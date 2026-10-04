"""Processing, analysis and visualization for the vLLM vs llama.cpp backend selection.

Layers (each only depends on the ones above it):

    paths, data.registry        where things are; what each configuration/campaign is
    data.telemetry              raw 1 s telemetry -> window aggregates
    data.processing             data/raw -> data/processed (run by `make data`)
    data.loaders                data/processed -> DataFrames for the notebooks
    analysis.*                  derived metrics and comparisons (no plotting)
    visualization.*             Plotly theme, figure builders and static export
"""

__version__ = "0.2.0"
