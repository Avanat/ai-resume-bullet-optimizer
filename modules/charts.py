"""Plotly chart builders (all return go.Figure; styling follows the Streamlit theme)."""
from __future__ import annotations

from typing import Dict, List, Optional

import plotly.graph_objects as go

PRIMARY, GOOD, WARN, BAD, MUTED = "#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#94A3B8"


def _color(v: float) -> str:
    return GOOD if v >= 75 else WARN if v >= 50 else BAD


def _base(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value), title={"text": title},
        number={"suffix": "%" if "Match" in title else ""},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": _color(value)},
               "steps": [{"range": [0, 50], "color": "rgba(239,68,68,.15)"},
                         {"range": [50, 75], "color": "rgba(245,158,11,.15)"},
                         {"range": [75, 100], "color": "rgba(16,185,129,.15)"}]}))
    return _base(fig, 260)


def radar(scores: Dict[str, float], title: str = "Resume score breakdown") -> go.Figure:
    labels, vals = list(scores), [round(v) for v in scores.values()]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], fill="toself",
                                    line=dict(color=PRIMARY), fillcolor="rgba(79,70,229,.25)"))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
                      showlegend=False)
    return _base(fig, 340)


def skills_gap_bar(jd_info: Dict[str, dict], matched: List[str], missing: List[str]) -> go.Figure:
    skills = (missing + matched)[::-1]
    vals = [jd_info[s]["mentions"] for s in skills]
    colors = [GOOD if s in matched else BAD for s in skills]
    fig = go.Figure(go.Bar(x=vals, y=skills, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: mentioned %{x}x in the job<extra></extra>"))
    fig.update_layout(title="Skills the job asks for  (green = on your resume, red = missing)",
                      xaxis_title="Mentions in job description")
    return _base(fig, max(260, 28 * len(skills) + 90))


def keyword_bar(keywords: List[dict]) -> go.Figure:
    kws = keywords[:15][::-1]
    fig = go.Figure(go.Bar(x=[k["weight"] for k in kws], y=[k["term"] for k in kws], orientation="h",
                           marker_color=[GOOD if k["in_resume"] else BAD for k in kws]))
    fig.update_layout(title="Top job keywords by importance (TF-IDF)", xaxis_title="Relative weight")
    return _base(fig, max(260, 26 * len(kws) + 90))


def bullet_relevance_bar(items: List[dict], max_items: int = 12) -> go.Figure:
    items = sorted(items, key=lambda x: x["relevance"])[-max_items:]
    labels = [(i["text"][:48] + "…") if len(i["text"]) > 48 else i["text"] for i in items]
    fig = go.Figure(go.Bar(x=[i["relevance"] for i in items], y=labels, orientation="h",
                           marker_color=[_color(i["relevance"]) for i in items]))
    fig.update_layout(title="How relevant each bullet is to the job", xaxis=dict(range=[0, 100], title="Relevance"))
    return _base(fig, max(260, 34 * len(items) + 90))


def before_after(before: List[float], after: List[float]) -> go.Figure:
    n = min(len(before), 15)
    idx = [f"#{i + 1}" for i in range(n)]
    fig = go.Figure([go.Bar(name="Original", x=idx, y=before[:n], marker_color=MUTED),
                     go.Bar(name="Optimized", x=idx, y=after[:n], marker_color=PRIMARY)])
    fig.update_layout(barmode="group", title="Bullet quality score: before vs after",
                      yaxis=dict(range=[0, 100], title="Score"), legend=dict(orientation="h", y=1.12))
    return _base(fig, 320)


def category_donut(categories: Dict[str, List[str]]) -> go.Figure:
    fig = go.Figure(go.Pie(labels=list(categories), values=[len(v) for v in categories.values()], hole=0.55))
    fig.update_layout(title="Your skills by category")
    return _base(fig, 320)


def score_bars(components: Dict[str, float]) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(components), y=[round(v) for v in components.values()],
                           marker_color=[_color(v) for v in components.values()],
                           text=[f"{round(v)}" for v in components.values()], textposition="outside"))
    fig.update_layout(title="Score by category", yaxis=dict(range=[0, 110]))
    return _base(fig, 320)
