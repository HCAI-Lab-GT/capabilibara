"""Interactive and static treemap visualizations for corpus composition."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd
import plotly.graph_objects as go

from dolma.distribution_report.data_loader import display_label
from dolma.distribution_report.style import (
    paper_layout,
    save_figure,
)

logger = logging.getLogger(__name__)

TREEMAP_SIZE_PX = 500

_TOPIC_COLORS = [
    "rgba(251,177,208,0.85)",
    "rgba(252,179,190,0.85)",
    "rgba(254,180,150,0.85)",
    "rgba(253,179,170,0.85)",
    "rgba(251,176,228,0.85)",
    "rgba(251,176,208,0.85)",
    "rgba(252,179,190,0.85)",
    "rgba(252,179,190,0.85)",
    "rgba(251,177,208,0.85)",
    "rgba(255,180,130,0.85)",
    "rgba(251,177,208,0.85)",
    "rgba(251,177,208,0.85)",
    "rgba(252,179,190,0.85)",
    "rgba(253,179,170,0.85)",
    "rgba(250,176,228,0.85)",
    "rgba(250,176,228,0.85)",
    "rgba(255,180,130,0.85)",
    "rgba(255,180,130,0.85)",
    "rgba(253,179,170,0.85)",
    "rgba(254,180,150,0.85)",
    "rgba(251,177,208,0.85)",
    "rgba(251,177,208,0.85)",
    "rgba(253,179,170,0.85)",
    "rgba(251,177,208,0.85)",
]

_FORMAT_COLORS = [
    "rgba(180,234,241,0.85)",
    "rgba(178,230,241,0.85)",
    "rgba(168,213,243,0.85)",
    "rgba(185,242,240,0.85)",
    "rgba(163,205,244,0.85)",
    "rgba(185,242,240,0.85)",
    "rgba(173,222,242,0.85)",
    "rgba(168,213,243,0.85)",
    "rgba(185,242,240,0.85)",
    "rgba(178,230,241,0.85)",
    "rgba(178,230,241,0.85)",
    "rgba(178,230,241,0.85)",
    "rgba(185,242,240,0.85)",
    "rgba(185,242,240,0.85)",
    "rgba(161,201,244,0.85)",
    "rgba(161,201,244,0.85)",
    "rgba(185,242,240,0.85)",
    "rgba(163,205,244,0.85)",
    "rgba(173,222,242,0.85)",
    "rgba(173,222,242,0.85)",
    "rgba(180,234,241,0.85)",
    "rgba(173,222,242,0.85)",
    "rgba(168,213,243,0.85)",
    "rgba(163,205,244,0.85)",
]


def _topic_colors(n: int) -> list[str]:
    return [_TOPIC_COLORS[i % len(_TOPIC_COLORS)] for i in range(n)]


def _format_colors(n: int) -> list[str]:
    return [_FORMAT_COLORS[i % len(_FORMAT_COLORS)] for i in range(n)]


def _static_treemap(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output_dir: Path,
    filename: str,
    formats: tuple[str, ...],
    colors: list[str] | None = None,
) -> None:
    sorted_df = df.sort_values(value_col, ascending=False)
    labels = [display_label(lbl) for lbl in sorted_df[label_col]]
    values = sorted_df[value_col].tolist()
    if colors is None:
        colors = _topic_colors(len(labels))

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=[""] * len(labels),
            values=values,
            textinfo="label+percent root",
            marker={"colors": colors},
            tiling={"packing": "squarify", "pad": 2},
        )
    )
    layout = paper_layout(title)
    layout["margin"] = {"l": 10, "r": 10, "t": 60, "b": 10}
    fig.update_layout(
        **layout,
        width=TREEMAP_SIZE_PX,
        height=TREEMAP_SIZE_PX,
    )
    save_figure(fig, output_dir, filename, formats, TREEMAP_SIZE_PX, TREEMAP_SIZE_PX)


def fig_treemap_topic(
    topic_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _static_treemap(
        topic_df,
        "label",
        "token_count_est",
        "Corpus Composition by Topic",
        output_dir,
        "fig_treemap_topic",
        formats,
    )


def fig_treemap_format(
    format_df: pd.DataFrame,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    _static_treemap(
        format_df,
        "label",
        "token_count_est",
        "Corpus Composition by Format",
        output_dir,
        "fig_treemap_format",
        formats,
        colors=_format_colors(len(format_df)),
    )


def _build_treemap_json(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    colors: list[str],
) -> list[dict]:
    return [
        {
            "name": display_label(cast(str, row[label_col])),
            "value": int(cast(int | float, row[value_col])),
            "color": colors[i],
        }
        for i, (_, row) in enumerate(df.iterrows())
    ]


def _build_joint_json(
    grid_df: pd.DataFrame, topics: list[str], formats: list[str]
) -> list[dict]:
    topic_idx = {lbl: i for i, lbl in enumerate(topics)}
    format_idx = {lbl: i for i, lbl in enumerate(formats)}
    entries = []
    for _, row in grid_df.iterrows():
        topic_label = cast(str, row["topic_label"])
        format_label = cast(str, row["format_label"])
        ti = topic_idx.get(topic_label)
        fi = format_idx.get(format_label)
        val = int(cast(int | float, row.get("token_count_est", 0)))
        if ti is not None and fi is not None and val > 0:
            entries.append({"t": ti, "f": fi, "w": val})
    return entries


def fig_treemap_interactive(
    topic_df: pd.DataFrame,
    format_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    topic_labels = topic_df["label"].tolist()
    format_labels = format_df["label"].tolist()
    tc = _topic_colors(len(topic_labels))
    fc = _format_colors(len(format_labels))

    topics_json = json.dumps(
        _build_treemap_json(topic_df, "label", "token_count_est", tc)
    )
    formats_json = json.dumps(
        _build_treemap_json(format_df, "label", "token_count_est", fc)
    )
    joint_json = json.dumps(_build_joint_json(grid_df, topic_labels, format_labels))

    html = _INTERACTIVE_TEMPLATE.replace("__TOPICS_JSON__", topics_json)
    html = html.replace("__FORMATS_JSON__", formats_json)
    html = html.replace("__JOINT_JSON__", joint_json)

    path = Path(output_dir) / "fig_treemap_interactive.html"
    path.write_text(html, encoding="utf-8")
    logger.info("Saved interactive treemap to %s", path)


_INTERACTIVE_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Corpus Treemap</title>
<link href="https://fonts.googleapis.com/css?family=Google+Sans|Noto+Sans|Castoro" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
body { font-family: 'Noto Sans', sans-serif; margin: 0; padding: 20px 40px;
       background: #fff; color: #333; }
h2 { text-align: center; font-family: 'Google Sans', sans-serif;
     font-size: 1.6rem; margin-bottom: 4px; }
p.hint { text-align: center; color: #888; font-size: 0.82rem; margin-top: 0; }
.container { display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;
             margin: 20px 0; }
.panel { display: flex; flex-direction: column; align-items: center; }
.panel-label { font-family: 'Castoro', serif; font-style: italic;
               font-size: 1.15rem; margin-bottom: 6px; }
.panel-label.topic { color: #c1615d; }
.panel-label.format { color: #3c7ab6; }
.treemap { width: 450px; height: 450px; }
</style>
</head>
<body>
<h2>Corpus Composition</h2>
<p class="hint">Areas show estimated token count per domain.
Hover on a topic to see the format breakdown, or vice versa.</p>
<div class="container">
  <div class="panel">
    <div class="panel-label topic">Topic Domains</div>
    <div id="topic-treemap" class="treemap"></div>
  </div>
  <div class="panel">
    <div class="panel-label format">Format Domains</div>
    <div id="format-treemap" class="treemap"></div>
  </div>
</div>
<script>
const T = __TOPICS_JSON__;
const F = __FORMATS_JSON__;
const J = __JOINT_JSON__;

function marginal(data) { return data.map(d => d.value); }

function conditional(joint, condType, idx, n) {
  const dist = new Array(n).fill(0);
  for (const e of joint) {
    if (condType === 't' && e.t === idx) dist[e.f] += e.w;
    if (condType === 'f' && e.f === idx) dist[e.t] += e.w;
  }
  const sum = dist.reduce((a, b) => a + b, 0);
  return sum > 0 ? dist.map(v => v / sum) : dist;
}

function mkTrace(data) {
  const total = data.reduce((s, x) => s + x.value, 0);
  return {
    type: 'treemap', parents: data.map(() => ''),
    labels: data.map(d => d.name), values: marginal(data),
    marker: { colors: data.map(d => d.color) },
    textinfo: 'label+percent root', sort: false,
    textfont: { color: 'black', size: 11 },
    tiling: { packing: 'squarify', flip: 'y', pad: 2, squarifyratio: 1 },
    hovertext: data.map(d => d.name+'<br>'+(d.value/total*100).toFixed(1)+'%'),
    hoverinfo: 'text', hoverlabel: { font: { color: 'black' } }
  };
}

const cfg = { displayModeBar: false };
const lay = { margin: {l:0,r:0,t:0,b:0}, width: 450, height: 450 };

Plotly.newPlot('topic-treemap', [mkTrace(T)], lay, cfg);
Plotly.newPlot('format-treemap', [mkTrace(F)], lay, cfg);

const tEl = document.getElementById('topic-treemap');
const fEl = document.getElementById('format-treemap');
const tMarg = marginal(T);
const fMarg = marginal(F);
let lockedTopic = null;
let lockedFormat = null;

tEl.on('plotly_hover', e => {
  if (lockedTopic !== null) return;
  Plotly.update('format-treemap',{values:[conditional(J,'t',e.points[0].pointNumber,F.length)]});
});
tEl.on('plotly_unhover', () => {
  if (lockedTopic !== null) return;
  Plotly.update('format-treemap', { values: [fMarg] });
});
tEl.on('plotly_treemapclick', e => {
  const idx = e.points[0].pointNumber;
  if (lockedTopic === idx) { lockedTopic = null; Plotly.update('format-treemap',{values:[fMarg]}); }
  else { lockedTopic = idx; Plotly.update('format-treemap',{values:[conditional(J,'t',idx,F.length)]}); }
  return false;
});

fEl.on('plotly_hover', e => {
  if (lockedFormat !== null) return;
  Plotly.update('topic-treemap',{values:[conditional(J,'f',e.points[0].pointNumber,T.length)]});
});
fEl.on('plotly_unhover', () => {
  if (lockedFormat !== null) return;
  Plotly.update('topic-treemap', { values: [tMarg] });
});
fEl.on('plotly_treemapclick', e => {
  const idx = e.points[0].pointNumber;
  if (lockedFormat === idx) { lockedFormat = null; Plotly.update('topic-treemap',{values:[tMarg]}); }
  else { lockedFormat = idx; Plotly.update('topic-treemap',{values:[conditional(J,'f',idx,T.length)]}); }
  return false;
});
</script>
</body>
</html>
"""

__all__ = [
    "fig_treemap_format",
    "fig_treemap_interactive",
    "fig_treemap_topic",
]
