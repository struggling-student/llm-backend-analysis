import plotly.graph_objects as go
import pytest

from llm_backend_analysis.visualization import export


def test_svg_normalization_is_stable():
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4], uid="t000"))
    a = export._normalize_svg(fig.to_image(format="svg"))
    b = export._normalize_svg(fig.to_image(format="svg"))
    assert a == b


def test_figure_names_must_be_snake_case(monkeypatch):
    monkeypatch.setenv("LBA_EXPORT", "0")
    with pytest.raises(ValueError):
        export.export_figure(go.Figure(), "Plot-Final2", source="x.ipynb", caption="")
