# src/utils/charts.py

import plotly.graph_objects as go
from src.utils.config import T


def base_layout(**overrides) -> dict:
    layout = dict(
        paper_bgcolor=T["paper_bg"],
        plot_bgcolor=T["bg"],
        font=dict(family=T["font"], color=T["text"], size=12),
        xaxis=dict(gridcolor=T["grid"], zeroline=False, showline=False,
                   tickfont=dict(size=11, color=T["text"])),
        yaxis=dict(gridcolor=T["grid"], zeroline=False, showline=False,
                   tickfont=dict(size=11, color=T["text"])),
        margin=dict(l=60, r=20, t=50, b=45),
        hoverlabel=dict(bgcolor=T["card_bg"], bordercolor=T["accent"],
                        font=dict(family=T["font"], size=12, color=T["text_bright"])),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=T["border"],
                    font=dict(color=T["text"])),
        hovermode="x unified",
    )
    layout.update(overrides)
    return layout


def linie(x, y, name: str, color: str, fill: bool = False,
          dash: str = "solid", width: int = 2) -> go.Scatter:
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    return go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color=color, width=width, dash=dash),
        fill="tozeroy" if fill else None,
        fillcolor=f"rgba({r},{g},{b},0.07)" if fill else None,
    )
