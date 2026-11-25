"""
One-off extraction: CIS barometer microdata -> data/psoe_2023_voters.csv

NOT part of `main.py train`. It reads the raw SPSS files kept outside this
repo (the GAD3 observatory) and writes the committed CSV that train.py
loads. Run it only to rebuild the dataset:

    uv run --with pyreadstat python data/build_dataset.py   (needs the observatory's raw files)

Scope decisions made here (see REPORT.md):
- Window: barometers with the 2023 general election as vote-recall anchor
  (Oct 2023 onwards). Every study in it is telephone (CATI), carries PESO
  and asks INTENCIONG / RECUVOTOG with the same wording, so the series is
  methodologically coherent. We keep the most recent FIRST_STUDY onwards.
- Population: respondents who recall voting PSOE in July 2023.
- Target: churn = 1 unless current vote intention is PSOE. Respondents with
  no declared intention (undecided / no answer) keep their row: they are
  the set the model allocates, never training data.
- Kept: CERCANIA (party closest to one's ideas). It is asked of everyone,
  so it is available for the undecided, and it is what the CIS's own
  "voto + simpatia" logic uses for them.
- Dropped at source: direct identifiers / quasi-identifiers (record id,
  municipality, province, interviewer) and variables that restate the
  intention or are only asked of the undecided (SIMPATIA, VOTOSIMG,
  preferred PM, alternative vote).
- Weights: each barometer is raked on its WHOLE sample before the PSOE voters
  are kept (data/raking.py, the observatory / TFG Layer 0 correction): recall
  of the 23J vote + sex x age + education, starting from the CIS weight.
  `weight` is that corrected weight; `weight_cis` is the CIS's own.
- Source files: only the package of record, Barómetro CIS/Datos/<year>/MD<n>.zip.
  Other copies exist on disk and are not identical (3567 differs).
"""

import argparse
import csv
import html
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

import raking as R

ROOT = Path(__file__).resolve().parent
OBS = Path.home() / "Desktop/GAD3/Observatorio Precision Encuestas"
CENSUS = OBS / "TFG/Datos/CIS/censo-cuestionario-2015-2026.csv"
RAW = OBS / "Barómetro CIS"

PSOE = 1  # code in RECUVOTOGR / INTENCIONGR (checked against labels below)

# Ordinal scales: kept as numeric codes (N.S./N.C. codes resolved in train.py)
NUMERIC = [
    "EDAD", "TAMUNI", "ECOPER", "ECOESP", "PROBVOTO", "ESCIDEOL",
    "VALORALIDERES_1", "VALORALIDERES_2", "VALORALIDERES_3", "VALORALIDERES_4",
    "PRACTICARELIG6", "INGRESHOG", "SINCERIDAD", "NIVELESTENTREV",
]
# Open-ended "main problem" answers (~60 categories). The CIS reuses codes
# with different meanings across this window (e.g. 90 = "depopulation" in
# early studies, "regional government management" later), so each answer is
# translated with its own study's labels and re-coded into one global table
# (codebook.json -> PROBLEM__values). Codes keep the CSV small.
PROBLEMS = [
    "PESPANNA1", "PESPANNA2", "PESPANNA3",
    "PPERSONAL1", "PPERSONAL2", "PPERSONAL3",
]
# Nominal variables: exported as their CIS text label
NOMINAL = [
    "CCAA", "SEXO", "ESTUDIOS", "RELIGION", "ECIVIL", "SITLAB", "CLASESOCIAL",
    "LIDERESCONOCE_1", "LIDERESCONOCE_2", "LIDERESCONOCE_3", "LIDERESCONOCE_4",
    "TIPO_TEL", "CERCANIA",
]


def find_sav(study: str, tmp: Path) -> Path:
    """The package of record (the one the observatory loaded), never another copy."""
    for z in (RAW / "Datos").glob(f"*/MD{study}.zip"):
        with zipfile.ZipFile(z) as zf:
            savs = [n for n in zf.namelist() if n.lower().endswith(".sav")]
            if savs:
                return Path(zf.extract(savs[0], tmp / study))
    raise FileNotFoundError(f"No Datos/*/MD{study}.zip")


def corrected_weights(df: pd.DataFrame, meta, study: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(CIS weight, raked weight, diagnostics) for the whole sample of one barometer."""
    info = R.TARGETS["studies"][study]
    label = lambda v: df[v].map(meta.value_labels[meta.variable_to_label[v]])  # noqa: E731
    axes = {
        "recuerdo": R.recall_axis(label("PARTICIPACIONG"), df["RECUVOTOG"]),
        "sexo_edad": R.sex_age_axis(label("SEXO"), df["EDAD"]),
        "estudios": R.education_axis(label("ESTUDIOS")),
    }
    w_cis = df[info["weight_variable"]].to_numpy(float)
    targets, tranches = R.demographic_targets(info["published"])
    targets["recuerdo"] = R.recall_target()
    raked = R.rake(axes, w_cis, targets)
    # Identity check: the same raking with the console's targets must give the
    # PSOE estimate the observatory has stored for this barometer.
    console = R.console_targets(info["month"]) | {"recuerdo": R.recall_target()}
    check = R.intention_share(df["INTENCIONG"], R.rake(axes, w_cis, console)["weights"], "psoe")
    stored = info["observatory_check"]["intencion_raking_psoe"]
    assert abs(check - stored) < 0.05, (study, check, stored)
    psoe = np.array(axes["recuerdo"], dtype=object) == "psoe"
    diag = {
        "weight_variable": info["weight_variable"], "published": info["published"], "targets_used": tranches,
        "sweeps": raked["sweeps"], "converged": raked["converged"], "trimmed": raked["trimmed"],
        "n": len(df), "n_ef": round(raked["n_ef"], 1), "deff": round(raked["deff"], 3),
        "psoe_recall_share_before": raked["axes"]["recuerdo"]["before"]["psoe"],
        "psoe_recall_share_after": raked["axes"]["recuerdo"]["after"]["psoe"],
        "psoe_factor_mean": round(float((raked["weights"][psoe] / w_cis[psoe]).mean()), 4),
        "psoe_factor_range": [round(float(x), 4) for x in np.quantile(raked["weights"][psoe] / w_cis[psoe], [0, 1])],
        "identity_check_psoe": [round(check, 4), round(stored, 4)],
        "axes": raked["axes"],
    }
    return w_cis, raked["weights"], diag


def main(first_study: int) -> None:
    census = list(csv.DictReader(CENSUS.open()))
    studies = [
        r for r in census
        if r["tipo_consola"] == "barometro" and int(r["estudio"]) >= max(first_study, 3423)
    ]
    frames, codebook, problem_codes, diagnostics = [], {}, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for r in studies:
            study = r["estudio"]
            df, meta = pyreadstat.read_sav(find_sav(study, Path(tmp)))
            w_cis, w_raked, diagnostics[study] = corrected_weights(df, meta, study)
            df = df.assign(_w_cis=w_cis, _w_raked=w_raked)
            for var in ("RECUVOTOGR", "INTENCIONGR"):
                label = meta.value_labels[meta.variable_to_label[var]][PSOE]
                assert label == "PSOE", (study, var, label)
            df = df[df["RECUVOTOGR"] == PSOE]
            out = pd.DataFrame({
                "study": int(study),
                "field_date": pd.to_datetime(dict(
                    year=2000 + df["IA_E1_ANYO"], month=df["IA_E1_MES"], day=df["IA_E1_DIA"]
                ), errors="coerce").dt.date,
                "weight": df["_w_raked"],      # corrected (raked) weight, see raking.py
                "weight_cis": df["_w_cis"],    # the CIS's own (PESO; PESO_V2 in 3528)
                "interview_minutes": df.get("IA_E3"),
            })
            labels_of = lambda v: {  # noqa: E731
                k: " ".join(html.unescape(t).split())
                for k, t in meta.value_labels.get(meta.variable_to_label.get(v), {}).items()
            }
            for v in NUMERIC:
                # Ordinal scales: codes are stable across the window (only the
                # label wording changes), so codes are kept as-is
                out[v] = df[v].astype("Int64")
                codebook.setdefault(v, meta.column_names_to_labels[v])
                codebook.setdefault(f"{v}__values", {str(int(k)): t for k, t in labels_of(v).items()})
            for v in PROBLEMS:
                text = df[v].map(labels_of(v))
                out[v] = text.map(lambda t: problem_codes.setdefault(t, len(problem_codes))
                                  if isinstance(t, str) else pd.NA).astype("Int64")
                codebook.setdefault(v, meta.column_names_to_labels[v])
            for v in NOMINAL:
                out[v] = df[v].map(labels_of(v)).fillna(df[v].astype("string"))
                codebook.setdefault(v, meta.column_names_to_labels[v])
            intent_labels = meta.value_labels[meta.variable_to_label["INTENCIONGR"]]
            # Destination is for the dashboard's "where do they go" view only:
            # it is the target restated and must never be used as a feature.
            out["destination"] = df["INTENCIONGR"].map(intent_labels)
            out["churn"] = (df["INTENCIONGR"] != PSOE).astype(int)
            frames.append(out)
            d = diagnostics[study]
            print(f"{study}: {len(out):5d} PSOE-2023 voters, churn {out.churn.mean():.1%} | raking: "
                  f"{d['sweeps']} sweeps, PSOE recall {d['psoe_recall_share_before']:.1f}% -> "
                  f"{d['psoe_recall_share_after']:.1f}%, deff {d['deff']}")

    data = pd.concat([f.dropna(axis=1, how="all") for f in frames], ignore_index=True)
    data.to_csv(ROOT / "psoe_2023_voters.csv", index=False)
    codebook["PROBLEM__values"] = {str(c): t for t, c in problem_codes.items()}
    (ROOT / "codebook.json").write_text(json.dumps(codebook, ensure_ascii=False, indent=1))
    (ROOT / "raking_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=1))
    print(f"\n{len(data)} rows x {data.shape[1]} cols, churn {data.churn.mean():.1%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--first-study", type=int, default=3505, help="Apr 2025 barometer")
    main(p.parse_args().first_study)
