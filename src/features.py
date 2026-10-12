"""Feature engineering. OWNER WRITES THIS FILE.

Contract (keep it, train.py and app.py depend on it):

    build_features(df: pd.DataFrame, fit: dict | None = None) -> tuple[pd.DataFrame, dict]

- `df` is a frame of raw rows (train, test, or a single SKU typed into the dashboard).
- `fit` is None when called on the TRAIN set: compute anything that must be learned from
  training data (frequency tables for supplier/dept, the list of top-N departments, the scaler)
  and return it as the second value.
- `fit` is that dict when called on TEST rows or dashboard input: reuse it, learn nothing.
- Returns a numeric DataFrame (all float, no NaN) with stable column order, and the fit dict.

Never read any column listed in src.data.LEAKY_COLUMNS.

Helpers below are a starting point; the assembly is yours.
"""
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FAMILY_KEYWORDS = {
    "GANCHO": ["GANCHO", "HOOK"],
    "PERCHA": ["PERCHA", "HANGER"],
    "ESTANTERIA": ["ESTANTE", "TRAMERIA", "RACK", "LARGUERO", "PARANTE", "REPISA"],
    "CHECKOUT": ["CHECK OUT", "CHECK-OUT", "CHECKOUT"],
    "VITRINA": ["VITRINA", "SHOWCASE"],
    "MANIQUI": ["MANIQUI", "MANNEQUIN", "BUSTO"],
    "EXHIBIDOR": ["EXHIBIDOR", "EXH.", "DISPLAY", "STAND"],
    "PORTAPRECIO": ["PORTA PRECIO", "PORTAPRECIO", "PRICE"],
    "CARRO": ["CARRO", "CARRETILLA", "CANASTA", "CESTA"],
    "CERRADURA": ["CERRADURA", "LLAVE", "CANDADO"],
}
MATERIAL_KEYWORDS = {
    "METAL": ["METAL", "NIQUEL", "NIQ.", "ZINC", "CROMO", "CROMADO", "ACERO", "HIERRO", "ALUMINIO", "SATIN"],
    "MADERA": ["MADERA", "CAOBA", "WOOD"],
    "ACRILICO": ["ACRILIC", "ACRYLIC"],
    "PLASTICO": ["PLAST", "PVC", "POLI"],
    "CRISTAL": ["CRISTAL", "VIDRIO", "GLASS"],
}
COLORS = ["BLANC", "NEGR", "GRIS", "AZUL", "ROJ", "VERDE", "NARANJ", "AMARILL", "CLEAR", "NAT"]

INCH_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:PLG|PULG|\"|'')", re.I)
MM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*MM", re.I)
MTS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:MTS|MT|M)\b", re.I)
DIMS_RE = re.compile(r"\d+(?:[.,]\d+)?\s*[xX]\s*\d+(?:[.,]\d+)?")


def _first_match(rx: re.Pattern, text: str) -> float:
    m = rx.search(text)
    return float(m.group(1).replace(",", ".")) if m else np.nan


def family(desc: str) -> str:
    d = desc.upper()
    for fam, kws in FAMILY_KEYWORDS.items():
        if any(k in d for k in kws):
            return fam
    return "OTHER"


def material(desc: str) -> str:
    d = desc.upper()
    for mat, kws in MATERIAL_KEYWORDS.items():
        if any(k in d for k in kws):
            return mat
    return "OTHER"


def size_cm(desc: str) -> float:
    """Best-effort single size in cm from inches / mm / metres. NaN if nothing found."""
    inch = _first_match(INCH_RE, desc)
    if not np.isnan(inch):
        return inch * 2.54
    mm = _first_match(MM_RE, desc)
    if not np.isnan(mm):
        return mm / 10
    mts = _first_match(MTS_RE, desc)
    if not np.isnan(mts):
        return mts * 100
    return np.nan


def code_prefix(code: str) -> str:
    m = re.match(r"^([A-Za-z]+)", str(code))
    return m.group(1).upper() if m else "NUM"


def _assemble(df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """Build the UNSCALED feature frame using the levels/tables learned in `fit`.
    Same logic for train, test and dashboard rows — only `fit` is learned, never here."""
    desc = df["Descripción"].astype(str)
    code = df["Código"]

    # Categorical labels per row. Prefixes/depts outside the train top-N collapse to OTHER
    # so train and test always produce the same set of columns.
    fam = desc.map(family)
    mat = desc.map(material)
    pref = code.map(code_prefix)
    pref = pref.where(pref.isin(fit["top_prefixes"]), "OTHER")
    dpt_raw = df["Dpto (ID)"].astype(str)
    dpt_top = dpt_raw.where(dpt_raw.isin(fit["top_depts"]), "OTHER")

    parts = []
    # One-hot blocks. reindex to the stored level list → fixed columns, 0.0 for absent levels.
    parts.append(pd.get_dummies(fam).reindex(columns=fit["family_levels"], fill_value=0).add_prefix("fam_"))
    parts.append(pd.get_dummies(mat).reindex(columns=fit["material_levels"], fill_value=0).add_prefix("mat_"))
    parts.append(pd.get_dummies(pref).reindex(columns=fit["prefix_levels"], fill_value=0).add_prefix("pref_"))
    parts.append(pd.get_dummies(dpt_top).reindex(columns=fit["dept_levels"], fill_value=0).add_prefix("dpt_"))

    # Frequency encoding: how often each ID appears in TRAIN. Unseen IDs → 0.
    # One number instead of 74 supplier dummies on 2k rows (see REPORT.md, Features).
    freq = pd.DataFrame(index=df.index)
    freq["freq_supplier"] = df["Suplidor (ID)"].astype(str).map(fit["freq_supplier"]).fillna(0.0)
    freq["freq_dpto"] = dpt_raw.map(fit["freq_dpto"]).fillna(0.0)
    freq["freq_area"] = df["Área (ID)"].astype(str).map(fit["freq_area"]).fillna(0.0)
    parts.append(freq)

    # Parsed + numeric. NaNs filled with the TRAIN median so the frame is all-float, no-NaN.
    num = pd.DataFrame(index=df.index)
    # Size enters as log1p: raw size_cm is heavy-tailed (a few multi-metre fixtures among
    # centimetre hooks) and that tail was the ill-conditioned direction the manual gradient
    #-descent loop crawled along, leaving it 0.56 off sklearn on this one coefficient.
    num["log1p_size_cm"] = np.log1p(desc.map(size_cm).fillna(fit["size_median"]))
    num["has_dims"] = desc.map(lambda t: float(bool(DIMS_RE.search(t))))
    num["has_color"] = desc.map(lambda t: float(any(c in t.upper() for c in COLORS)))
    num["n_words"] = desc.str.split().map(len).astype(float)
    # Dropped on purpose: Empaque (1 for 99.7% of SKUs, so any other value lands many
    # standard deviations out) and log(Costo) (no AUC gain in the ablation).
    parts.append(num)

    return pd.concat(parts, axis=1).astype(float)


def build_features(df: pd.DataFrame, fit: dict | None = None) -> tuple[pd.DataFrame, dict]:
    if fit is None:
        fit = {}
        desc = df["Descripción"].astype(str)
        code = df["Código"]
        fit["family_levels"] = list(FAMILY_KEYWORDS) + ["OTHER"]
        fit["material_levels"] = list(MATERIAL_KEYWORDS) + ["OTHER"]
        fit["top_prefixes"] = code.map(code_prefix).value_counts().head(10).index.tolist()
        fit["prefix_levels"] = fit["top_prefixes"] + ["OTHER"]
        fit["top_depts"] = df["Dpto (ID)"].astype(str).value_counts().head(8).index.tolist()
        fit["dept_levels"] = fit["top_depts"] + ["OTHER"]
        fit["freq_supplier"] = df["Suplidor (ID)"].astype(str).value_counts().to_dict()
        fit["freq_dpto"] = df["Dpto (ID)"].astype(str).value_counts().to_dict()
        fit["freq_area"] = df["Área (ID)"].astype(str).value_counts().to_dict()
        fit["size_median"] = float(np.nanmedian(desc.map(size_cm).to_numpy(dtype=float)))

        raw = _assemble(df, fit)
        fit["columns"] = list(raw.columns)
        fit["scaler"] = StandardScaler().fit(raw.to_numpy())
    else:
        raw = _assemble(df, fit).reindex(columns=fit["columns"], fill_value=0.0)

    X = pd.DataFrame(fit["scaler"].transform(raw.to_numpy()), columns=fit["columns"], index=df.index)
    return X, fit
