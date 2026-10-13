# pandas only here, at the display boundary

import polars as pl
import polars.selectors as cs
from IPython.display import display
from pandas.io.formats.style import Styler

from src.config import CONFIG


def style_df(
    df: pl.DataFrame,
    precision: int = 2,
    caption: str | None = None,
) -> Styler:
    date_formats = {
        **{c: "{:%Y-%m-%d}" for c in df.select(cs.date()).columns},
        **{c: "{:%Y-%m-%d %H:%M}" for c in df.select(cs.datetime()).columns},
    }
    styler = (
        df.to_pandas()
        .style.background_gradient(cmap=CONFIG.plot.table_cmap)
        .format(
            date_formats,
            precision=precision,
            thousands=",",
            na_rep="",
        )
    )
    return styler.set_caption(caption) if caption else styler


def show_df(
    df: pl.DataFrame,
    precision: int = 2,
    caption: str | None = None,
) -> None:
    display(style_df(df, precision, caption))
