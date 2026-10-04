"""Notebook presentation helpers: computed Markdown, readable tables, table exports and claim checks.

Numbers quoted in notebook prose are produced by these helpers from the data, never typed.
A *claim* pairs a sentence used in the interpretation with the boolean test that supports
it; `claims_table` shows whether each still holds, so re-running a notebook on new data
(e.g. vLLM INT8) flags interpretation text that needs revisiting.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import pandas as pd
from IPython.display import HTML, Markdown, display

from . import paths


def show(text: str) -> None:
    display(Markdown(text))


def x(v: float, nd: int = 1) -> str:
    """Ratio as '5.4×'."""
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}×"


def pct(v: float, nd: int = 0, signed: bool = True) -> str:
    """Fraction as '+18%' (signed) or '18%'."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v * 100:+.{nd}f}%" if signed else f"{v * 100:.{nd}f}%"


def num(v: float, nd: int = 0) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{nd}f}"


def rng(values: pd.Series, formatter=num, **kw) -> str:
    """'lo–hi' of a series with the given formatter (a single value if lo == hi)."""
    v = pd.Series(values).dropna()
    if v.empty:
        return "–"
    lo, hi = formatter(v.min(), **kw), formatter(v.max(), **kw)
    return lo if lo == hi else f"{lo}–{hi}"


def _stable_uuid(df: pd.DataFrame) -> str:
    """Deterministic table id (pandas uses a random one), so re-executed notebooks diff cleanly."""
    return "t" + hashlib.sha1(pd.util.hash_pandas_object(df.astype(str), index=False).values.tobytes()
                              + "|".join(map(str, df.columns)).encode()).hexdigest()[:10]


def table(df: pd.DataFrame, formats: dict[str, str] | None = None, caption: str | None = None,
          align: str = "right"):
    """Styled, index-free table for notebook display (columns renamed by the caller)."""
    sty = df.style.set_uuid(_stable_uuid(df)).hide(axis="index").format(na_rep="–", escape="html").format(formats or {}, na_rep="–")
    sty = sty.set_table_styles([{"selector": "th", "props": "text-align: left; font-weight: 600;"},
                                {"selector": "td", "props": f"text-align: {align}; padding: 2px 10px;"}])
    if caption:
        sty = sty.set_caption(caption)
    return HTML(sty.to_html())  # plain HTML: the Styler's text repr contains a memory address


def export_table(df: pd.DataFrame, name: str, *, topic: str = "backend_selection", float_format: str = "%.6g") -> None:
    """Write a derived table to results/tables/<topic>/<name>.csv (deterministic)."""
    out = paths.TABLES / topic
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False, float_format=float_format, lineterminator="\n")


@dataclass
class Claim:
    statement: str
    holds: bool
    evidence: str = ""


def claims_table(claims: list[Claim]):
    df = pd.DataFrame([{"Holds": "✓" if c.holds else "✗ revisit", "Statement": c.statement,
                        "Evidence (computed)": c.evidence} for c in claims])
    return HTML(df.style.set_uuid(_stable_uuid(df)).hide(axis="index").format(escape="html")
                .set_properties(**{"text-align": "left"}).to_html())
