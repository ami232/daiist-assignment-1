from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "psoe_2023_voters.csv"
CODEBOOK = ROOT / "data" / "codebook.json"

TARGET = "churn"

# No declared intention: these 2023 PSOE voters are what the model allocates.
# They never enter training or evaluation (they have no label).
UNDECIDED = {"No sabe todavía", "N.C."}

# Chronological split by barometer (study number = field month). The
# estimate for each month is made with a model fitted on past months.
TRAIN_STUDIES = range(3505, 3541)   # Apr 2025 - Jan 2026
VAL_STUDIES = range(3541, 3558)     # Feb - Apr 2026
TEST_STUDIES = range(3558, 3600)    # May - Sep 2026

LEADERS = {"sanchez": 1, "feijoo": 2, "diaz": 3, "abascal": 4}

# National "main problem" (PESPANNA1-3) answers -> themes. First match wins,
# so the more specific themes come first (e.g. "ocupación de viviendas" is a
# security frame, not a housing-affordability one).
PROBLEM_THEMES = [
    ("none_dk", ["n.s.", "n.c.", "ninguno"]),
    ("security_immigration", ["inmigración", "inseguridad ciudadana", "ocupación de viviendas",
                              "marruecos", "drogas", "terrorismo"]),
    ("housing", ["vivienda", "hipotecas", "desahucios"]),
    ("extremism", ["extremismos", "odio", "crispación"]),
    ("politics", ["polític", "partidos", "gobierno", "corrupción", "acuerdos", "democracia",
                  "justicia", "monarquía", "amnistía", "medios de comunicación", "fraude"]),
    ("economy_jobs", ["empleo", "paro", "económic", "impuestos", "autónomos", "productivo",
                      "energétic", "bancos", "reforma laboral"]),
    ("public_services", ["sanidad", "educación", "servicios públicos", "recortes", "pensiones",
                         "infraestructuras"]),
    ("social_equality", ["desigualdades", "género", "mujer", "juventud", "social", "racismo",
                         "climático", "medio ambiente"]),
    ("territorial", ["autonomía", "nacionalismos", "cataluña", "vaciada"]),
]
THEMES = [t for t, _ in PROBLEM_THEMES] + ["other"]

# Party closest to one's ideas (CERCANIA) -> blocs. ~55 labels, most with a
# handful of respondents, so they are grouped by where a vote could go.
CLOSEST_PARTY = {
    "psoe": ["PSOE"],
    "left": ["Sumar", "Podemos", "IU", "Unidas Podemos", "Más Madrid", "Más País", "Compromís",
             "Podemos-IU-AV", "En Comú Podem", "Comuns Sumar", "Coalición de izquierdas",
             "En Común-Unidas Podemos", "EQUO", "Los Verdes", "PCPE"],
    "regional": ["ERC", "EH Bildu", "BNG", "Junts", "EAJ-PNV", "CCa", "CUP", "Geroa Bai", "CHA",
                 "MÉS (PSM-Entesa)", "PRC", "UPL", "Nueva Canarias", "Adelante Andalucía",
                 "Andalucía Por Sí", "PAR", "Drago Canarias Verdes", "PI (Proposta per les Illes)",
                 "Soria Ya", "Jaén Merece Más", "Partido Riojano", "CDC", "PdeCat", "FAC"],
    "right": ["PP", "VOX", "Ciudadanos", "Se Acabó la Fiesta", "UPyD", "UPN", "Aliança Catalana",
              "Falange Española de las JONS"],
    "none": ["Ninguno"],
    "no_answer": ["N.S.", "N.C."],
}

# --- Feature blocks, in the order they are added in the ablation -----------
# Each block is (numeric, binary, categorical) column lists.
BLOCKS: dict[str, tuple[list[str], list[str], list[str]]] = {
    # Raw attitudes: "how do you rate each leader", turnout intent, ideology,
    # economy. N.S./N.C. become NaN and are median-imputed downstream.
    "core": (
        [f"rate_{l}" for l in LEADERS] + ["probvoto", "ideology", "eco_spain", "eco_personal"],
        [], [],
    ),
    # Not answering is itself information (disengagement), so it gets its
    # own flag instead of disappearing into the median.
    "dont_know_flags": (
        [], [f"rate_{l}_dk" for l in LEADERS] + ["probvoto_dk", "ideology_dk"], [],
    ),
    # Churn is relative: rating Sánchez 5 is safe if every rival gets a 2.
    "competitive_gap": (["gap_best_rival"], ["sanchez_top"], []),
    # Which party they feel closest to. Asked of everyone, undecided included:
    # it is what the CIS's own "vote + sympathy" leans on for them.
    "closest_party": ([], [], ["closest_group"]),
    # Non-linear shapes a linear model can't draw from the raw scales:
    # centrists churn most (U-shape in ideology); a "10" on turnout is a
    # qualitatively different voter from a "9".
    "ideology_turnout_shape": ([], ["probvoto_certain", "probvoto_low"], ["ideology_band"]),
    "problem_themes": ([], [f"problem_{t}" for t in THEMES], []),
    "sociodemographics": (
        ["town_size", "income_rank"],
        ["income_dk", "female"],
        ["age_band", "education", "religion", "work_status", "social_class", "region",
         "marital_status"],
    ),
    # Added in train.ipynb after k-means is fitted on the training months.
    "segment": ([], [], ["segment"]),
}

# Inputs of the k-means voter segmentation: the attitude block only, so the
# segments read as political profiles rather than demographic ones.
SEGMENT_INPUTS = [f"rate_{l}" for l in LEADERS] + ["ideology", "probvoto", "eco_spain", "eco_personal"]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA)


def is_declared(raw: pd.DataFrame) -> pd.Series:
    return ~raw["destination"].isin(UNDECIDED)


def split_of(study: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [study.isin(TRAIN_STUDIES), study.isin(VAL_STUDIES), study.isin(TEST_STUDIES)],
            ["train", "val", "test"], default="unused",
        ),
        index=study.index,
    )


def _scale(s: pd.Series, lo: int, hi: int) -> pd.Series:
    """Keep codes inside the valid range; N.S./N.C./N.P. codes -> NaN."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(lo, hi))


def _theme(label: str | float) -> str | None:
    if not isinstance(label, str):
        return None
    low = label.lower()
    for theme, keys in PROBLEM_THEMES:
        if any(k in low for k in keys):
            return theme
    return "other"


def _problem_labels() -> dict[int, str]:
    values = json.loads(CODEBOOK.read_text())["PROBLEM__values"]
    return {int(k): v for k, v in values.items()}


def _group(s: pd.Series, mapping: dict[str, str], default: str = "other") -> pd.Series:
    return s.map(lambda v: next((g for k, g in mapping.items() if isinstance(v, str) and k in v), default))


def engineer(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw CIS columns -> engineered features (every block's columns)."""
    f = pd.DataFrame(index=raw.index)

    # core + don't-know flags
    for name, i in LEADERS.items():
        f[f"rate_{name}"] = _scale(raw[f"VALORALIDERES_{i}"], 1, 10)
        f[f"rate_{name}_dk"] = f[f"rate_{name}"].isna().astype(int)
    f["probvoto"] = _scale(raw["PROBVOTO"], 0, 10)
    f["probvoto_dk"] = f["probvoto"].isna().astype(int)
    f["ideology"] = _scale(raw["ESCIDEOL"], 1, 10)
    f["ideology_dk"] = f["ideology"].isna().astype(int)
    # CIS codes 1 = very good ... 5 = very bad; flip so higher = better
    f["eco_spain"] = 6 - _scale(raw["ECOESP"], 1, 5)
    f["eco_personal"] = 6 - _scale(raw["ECOPER"], 1, 5)

    # competitive gap (NaN-safe: a rival nobody rated doesn't count)
    rivals = f[["rate_feijoo", "rate_diaz", "rate_abascal"]].max(axis=1)
    f["gap_best_rival"] = f["rate_sanchez"] - rivals
    f["sanchez_top"] = (f["gap_best_rival"] >= 0).astype(int)

    # closest party
    bloc = {label: g for g, labels in CLOSEST_PARTY.items() for label in labels}
    f["closest_group"] = raw["CERCANIA"].map(bloc).fillna("other")

    # shapes
    f["ideology_band"] = pd.cut(
        f["ideology"], [0, 2, 4, 6, 10], labels=["left_1-2", "centre_left_3-4", "centre_5-6", "right_7-10"]
    ).astype(object).fillna("no_answer")
    f["probvoto_certain"] = (f["probvoto"] == 10).astype(int)
    f["probvoto_low"] = (f["probvoto"] <= 4).astype(int)

    # problem themes: multi-hot over the three national-problem mentions
    labels = _problem_labels()
    themes = pd.concat(
        [raw[f"PESPANNA{i}"].map(lambda c: _theme(labels.get(int(c))) if pd.notna(c) else None)
         for i in (1, 2, 3)],
        axis=1,
    )
    for t in THEMES:
        f[f"problem_{t}"] = themes.eq(t).any(axis=1).astype(int)

    # sociodemographics
    f["town_size"] = _scale(raw["TAMUNI"], 1, 7)
    f["income_rank"] = 7 - _scale(raw["INGRESHOG"], 1, 6)   # 1 = lowest ... 6 = highest
    f["income_dk"] = f["income_rank"].isna().astype(int)
    f["female"] = (raw["SEXO"] == "Mujer").astype(int)
    f["age_band"] = pd.cut(raw["EDAD"], [0, 29, 44, 64, 120], labels=["18-29", "30-44", "45-64", "65+"]).astype(str)
    f["education"] = raw["ESTUDIOS"].where(raw["ESTUDIOS"].isin(
        ["Superiores", "F.P.", "Secundaria 2ª etapa", "Secundaria 1ª etapa", "Primaria", "Sin estudios"]
    ), "other")
    f["religion"] = _group(raw["RELIGION"], {
        "Católico/a practicante": "practising_catholic", "Católico/a no practicante": "nonpractising_catholic",
        "otra religión": "other_religion", "Ateo": "non_believer", "Agnóstico": "non_believer",
        "Indiferente": "non_believer",
    })
    f["work_status"] = _group(raw["SITLAB"], {
        "Trabaja": "working", "Jubilado": "retired", "Pensionista": "retired",
        "En paro": "unemployed", "Estudiante": "student",
    })
    f["social_class"] = _group(raw["CLASESOCIAL"], {
        "Clase alta": "upper", "media-alta": "upper", "media-media": "middle", "media-baja": "lower_middle",
        "trabajadora": "working_low", "Clase baja": "working_low", "pobre": "working_low",
        "Proletariado": "working_low", "abajo": "working_low", "Excluidos": "working_low",
        "gente común": "working_low",
    })
    f["region"] = raw["CCAA"].fillna("other")
    f["marital_status"] = raw["ECIVIL"].fillna("other")
    return f


def block_columns(blocks: list[str]) -> tuple[list[str], list[str], list[str]]:
    num, binary, cat = [], [], []
    for b in blocks:
        n, bi, c = BLOCKS[b]
        num += n
        binary += bi
        cat += c
    return num, binary, cat
