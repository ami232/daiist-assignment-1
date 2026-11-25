"""
Correction of each CIS barometer before anything is learned from it.

This is the correction used in the GAD3 observatory and in my TFG ("Layer 0",
`intencion_raking`), ported from `backend/observatorio/cis/correccion.py`
(`rastrillar`, `ejes_de`, `objetivo_recuerdo`) and `tfg/capa0.py`
(`objetivos_a_fecha`). Targets are in `data/raking_targets.json`.

Why: the CIS sample over-represents 2023 PSOE voters (about +6 points against
the 23J result, growing over time), and the CIS weight `PESO`, which only fixes
demographics, makes that worse. So each barometer is re-weighted, edition by
edition, on the WHOLE sample:

- start from the CIS weight (the variable its `WEIGHT BY` names: `PESO_V2` in 3528);
- rake (iterative proportional fitting) on three axes at once:
    recall of the 2023 vote   -> official 23J result (parties >= 1%, the rest pooled)
    sex x age (12 cells)      -> INE resident population 18+
    education (3 bands)       -> INE EPA population 20+
  with the demographic targets that were already PUBLISHED when the barometer
  came out (no future census data);
- each axis only moves people who have a category on it (non-voters and
  "don't remember" are not on the recall axis);
- the change is trimmed to [0.2, 5] times the CIS weight, weights renormalised
  to mean 1, stop when every margin is within 0.05 points (max 50 sweeps).

Only then are the 2023 PSOE voters kept. Their weights still differ by sex,
age and education, so the correction does not cancel inside the subset.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

TARGETS = json.loads((Path(__file__).resolve().parent / "raking_targets.json").read_text(encoding="utf-8"))

TRIM = (0.2, 5.0)
MIN_AXIS = 100          # cases with a category for an axis to be used
TOLERANCE = 0.05        # percentage points
OTHERS = "_otros"
AGE_BANDS = ((18, 24, "18-24"), (25, 34, "25-34"), (35, 44, "35-44"), (45, 54, "45-54"),
             (55, 64, "55-64"), (65, 200, "65+"))

# Modern CIS party codes (studies 3380 onwards) of the parties with >= 1% on 23J
# (`config/cis-partidos.csv`, column partido_final). Any other party code is
# pooled into OTHERS; technical codes (blank, null, didn't vote, N.S., N.C.)
# have no recall category.
PARTY_CODES = {1: "psoe", 2: "pp", 3: "vox", 21: "sumar", 205: "sumar", 815: "sumar", 912: "sumar",
               1605: "sumar", 901: "erc", 902: "junts", 907: "junts", 913: "junts", 1601: "pnv",
               1602: "bildu"}
TECHNICAL = {8996, 9977, 9997, 9998, 9999, 9995, 9996, 0}


def _normalise(text) -> str:
    if not isinstance(text, str):
        return ""
    s = unicodedata.normalize("NFD", text.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _voted(label: str) -> bool | None:
    """PARTICIPACIONG label -> voted (True), didn't / couldn't (False), unknown (None)."""
    s = re.sub(r"^no leer ", "", _normalise(label))
    if re.match(r"^(voto|fue a votar y voto|voto por correo)$", s):
        return True
    if re.match(r"^(no voto|prefirio no votar|no fue a votar|fue a votar pero no pudo|no tenia edad|no tenia derecho)", s):
        return False
    return None


def recall_axis(participation_labels, recall_codes) -> list[str | None]:
    """2023 recall as the raking sees it: a party id, OTHERS, or None (not on the axis)."""
    own = own_parties()
    out = []
    for label, code in zip(participation_labels, recall_codes):
        if _voted(label) is not True or code != code or int(code) in TECHNICAL:
            out.append(None)
            continue
        party = PARTY_CODES.get(int(code))
        out.append(party if party in own else OTHERS)
    return out


def sex_age_axis(sex_labels, ages) -> list[str | None]:
    out = []
    for s, a in zip(sex_labels, ages):
        s = _normalise(s)
        sex = "hombre" if s.startswith("hombre") else "mujer" if s.startswith("mujer") else None
        band = next((b for lo, hi, b in AGE_BANDS if a == a and 18 <= a <= 110 and lo <= a <= hi), None)
        out.append(f"{sex}:{band}" if sex and band else None)
    return out


EDUCATION = (("sin_estudios", r"^sin estudios"), ("primaria", r"^primaria"), ("secundaria1", r"^secundaria 1"),
             ("secundaria2", r"^secundaria 2"), ("fp", r"^f ?p$|^formacion profesional"),
             ("superiores", r"^superiores"))


def education_axis(education_labels) -> list[str | None]:
    eq = TARGETS["education_equivalence"]
    out = []
    for label in education_labels:
        s = _normalise(label)
        cat = next((c for c, pat in EDUCATION if re.search(pat, s)), None)
        out.append(eq.get(cat) if cat else None)
    return out


def own_parties() -> set[str]:
    official = {p: v for p, v in TARGETS["recall_2023_official"].items() if p != OTHERS}
    return {p for p, v in official.items() if v >= TARGETS["own_party_threshold_pct"]}


def recall_target() -> dict[str, float]:
    official = TARGETS["recall_2023_official"]
    own = own_parties()
    target = {p: float(official[p]) for p in own}
    target[OTHERS] = max(0.0, 100.0 - sum(target.values()))
    return target


def demographic_targets(published: str) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """TFG rule: the latest tranche already published on the day the barometer's
    file was published. An axis with no published tranche is left out."""
    targets, used = {}, {}
    for axis, spec in TARGETS["tfg_as_of"].items():
        live = [t for t in spec["tranches"] if t["publicado"] <= published]
        if live:
            t = max(live, key=lambda t: t["referencia"])
            targets[axis], used[axis] = dict(t["objetivo"]), f'{t["tabla"]} ({t["referencia"]})'
    return targets, used


def console_targets(month: str) -> dict[str, dict[str, float]]:
    """Observatory console rule (only for the identity check against the stored estimate)."""
    out = {}
    for axis, spec in TARGETS["console"].items():
        tranches = [t for t in spec["tranches"] if t["desde"] <= month] or spec["tranches"][:1]
        out[axis] = dict(max(tranches, key=lambda t: t["desde"])["objetivo"])
    return out


def kish(w: np.ndarray) -> tuple[float, float]:
    w = w[np.isfinite(w) & (w > 0)]
    n_ef = float(w.sum() ** 2 / (w ** 2).sum())
    return n_ef, float(len(w) / n_ef)


def rake(axes: dict[str, list], weights: np.ndarray, targets: dict[str, dict[str, float]],
         trim=TRIM, max_sweeps: int = 50, tolerance: float = TOLERANCE) -> dict:
    """Iterative proportional fitting, as `correccion.rastrillar`."""
    w0 = np.asarray(weights, dtype=float)
    prepared, skipped = [], {}
    for name, values in axes.items():
        target = {k: float(v) for k, v in (targets.get(name) or {}).items() if v is not None}
        if not target:
            skipped[name] = "no target"
            continue
        cat = np.asarray(values, dtype=object)
        inside = np.array([x is not None and str(x) in target for x in cat], dtype=bool)
        present = {str(x) for x in cat[inside]}
        target = {k: v for k, v in target.items() if k in present}
        if int(inside.sum()) < MIN_AXIS or len(target) < 2:
            skipped[name] = "too few cases or categories"
            continue
        total = sum(target.values())
        prepared.append((name, cat, inside, {k: 100.0 * v / total for k, v in target.items()}))

    w = w0.copy()
    sweeps, trimmed, converged = 0, 0, False
    for sweeps in range(1, max_sweeps + 1):
        worst = 0.0
        for _, cat, inside, target in prepared:
            base = float(w[inside].sum())
            for k, v in target.items():
                m = inside & (cat == k)
                current = float(w[m].sum())
                if current <= 0:
                    continue
                worst = max(worst, abs(100.0 * current / base - v))
                w[m] *= (base * v / 100.0) / current
        ratio = np.divide(w, w0, out=np.ones_like(w), where=w0 > 0)
        trimmed = int(((ratio < trim[0]) | (ratio > trim[1])).sum())
        w = np.where(w0 > 0, w0 * np.clip(ratio, *trim), w)
        w *= len(w) / float(w.sum())
        if worst < tolerance:
            converged = True
            break

    def shares(cat, inside, weights):
        base = weights[inside].sum()
        return {k: round(100 * float(weights[inside & (cat == k)].sum() / base), 3) for k in sorted(set(cat[inside]))}

    n_ef, deff = kish(w)
    return {"weights": w, "sweeps": sweeps, "converged": converged, "trimmed": trimmed, "n_ef": n_ef, "deff": deff,
            "skipped": skipped,
            "axes": {name: {"target": {k: round(v, 3) for k, v in t.items()}, "before": shares(c, i, w0),
                            "after": shares(c, i, w)} for name, c, i, t in prepared}}


def intention_share(intention_codes, weights, party: str) -> float:
    """Share of `party` over valid vote (CIS technical codes out, 'other party' in),
    as `estimadores.reparto(..., FUERA_VALIDO)`: used for the identity check."""
    codes = np.asarray(intention_codes, dtype=float)
    valid = ~np.isnan(codes) & ~np.isin(codes, list(TECHNICAL))
    is_party = np.array([PARTY_CODES.get(int(c)) == party if c == c else False for c in codes])
    return 100 * float(weights[valid & is_party].sum() / weights[valid].sum())
