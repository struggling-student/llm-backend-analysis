"""Backend tuning selection rule, declared before the campaign.

Per configuration: choose the variant with the highest steady total tokens/s at C=16,
unless its median TPOT at C=1 is more than 5% worse than the default variant's; then keep
the default. The default variant of each configuration is in configurations.toml.
The choice is copied by hand into configs/experiment.toml [tuned.*], so the harness
config stays the single reviewed source for what the campaign runs.
"""

from __future__ import annotations

import pandas as pd

from ..data import registry

SELECTION_CONCURRENCY = 16
BASELINE_CONCURRENCY = 1
MAX_TPOT_REGRESSION = 0.05


def select_variants(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, cid), g in runs.groupby(["model", "configuration_id"], sort=False):
        sel = g[(g["concurrency"] == SELECTION_CONCURRENCY) & g["total_tok_s"].notna()].set_index("variant")
        base = g[(g["concurrency"] == BASELINE_CONCURRENCY) & g["tpot_s_p50"].notna()].set_index("variant")
        if sel.empty:
            continue
        default = registry.configurations()[cid].tuning_default_variant
        best = sel["total_tok_s"].idxmax()
        reason = f"highest C={SELECTION_CONCURRENCY} throughput"
        if default in base.index and best in base.index and \
                base.loc[best, "tpot_s_p50"] > (1 + MAX_TPOT_REGRESSION) * base.loc[default, "tpot_s_p50"]:
            reason = f"{best} worsens C=1 TPOT by > {MAX_TPOT_REGRESSION:.0%}; default kept"
            best = default
        rows.append({"model": model, "configuration_id": cid, "default_variant": default, "chosen_variant": best,
                     "reason": reason, "chosen_total_tok_s": sel.loc[best, "total_tok_s"],
                     "default_total_tok_s": sel["total_tok_s"].get(default),
                     "gain_over_default": sel.loc[best, "total_tok_s"] / sel["total_tok_s"].get(default) - 1
                     if default in sel.index else float("nan"),
                     "variants_tried": ", ".join(sorted(sel.index))})
    return pd.DataFrame(rows)
