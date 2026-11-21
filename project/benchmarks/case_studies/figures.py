import numpy as np, warnings, os, json
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analyse import summarise

RED, AMBER, BLUE, INK = "#C8452C", "#D9A441", "#1F5FA8", "#22262B"
NAMES = {"births": ("US daily births", "births per day", "SeasonalNaive", "D", "1984-12-31"),
         "river": ("Saugeen River daily flow", "m³/s", "Naive", "D", None),
         "power": ("Victoria hourly electricity demand", "MW", "SeasonalNaive", "h", None)}
LAYOUT = dict(template="plotly_white", font=dict(family="Helvetica, Arial, sans-serif", size=14, color=INK),
              margin=dict(l=70, r=30, t=70, b=60), legend=dict(orientation="h", y=-0.18, x=0))
os.makedirs("out", exist_ok=True)
rows = []

def band(fig, x, lo, hi, color, name, row=None, col=None, opacity=0.25):
    kw = {} if row is None else dict(row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=x, y=lo, line=dict(width=0), fill="tonexty", name=name,
                             fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{opacity})"), **kw)

for series, model in [("births", "snaive"), ("river", "snaive"), ("power", "snaive")]:
    yt, F, res, ok, out, H = summarise(series, model, verbose=False)
    title, unit, mname, freq, _ = NAMES[series]
    n = int(ok[:, 0].sum())
    for k in ("statsforecast default (2 windows)", "statsforecast 10 windows", "pyconformal multi-step split", "pyconformal multi-step split, window 200", "pyconformal AcMCP"):
        d = out[k]; rows.append((title, mname, H, n, k, d["cov"], d["covh"].min(), d["width"]))
    # 1. coverage by horizon
    fig = go.Figure()
    hs = np.arange(1, H + 1)
    for k, c, dash in (("statsforecast default (2 windows)", RED, "solid"), ("statsforecast 10 windows", AMBER, "solid"), ("pyconformal multi-step split", BLUE, "solid")):
        fig.add_trace(go.Scatter(x=hs, y=100 * out[k]["covh"], mode="lines+markers", name=k, line=dict(color=c, width=3, dash=dash)))
    fig.add_hline(y=90, line=dict(color=INK, dash="dash", width=1.5), annotation_text="stated level: 90%", annotation_position="top left")
    fig.update_layout(**LAYOUT, title=f"{title}: coverage of 90% intervals by forecast horizon", yaxis=dict(title="coverage (%)", range=[50, 100]),
                      xaxis=dict(title="steps ahead", dtick=1 if H <= 12 else 4), width=900, height=480)
    fig.write_image(f"out/{series}_coverage.png", scale=2); fig.write_html(f"out/{series}_coverage.html", include_plotlyjs="cdn")
    # 2. the intervals on a stretch, at the longest horizon
    h = H - 1
    stretch = 12 * H if series != "power" else 7 * 24
    idx = np.flatnonzero(ok[:, h])[-stretch:]
    x = np.arange(len(idx))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("statsforecast, default settings", "pyconformal, same forecasts"))
    for r, (k, c) in enumerate((("statsforecast default (2 windows)", RED), ("pyconformal multi-step split", BLUE)), 1):
        lo, hi = res[k][0][idx, h], res[k][1][idx, h]
        if series == "river":
            lo = np.maximum(lo, 0.0)           # a flow cannot be negative; clipping does not change coverage
        band(fig, x, lo, hi, c, "90% interval", r, 1)
        fig.add_trace(go.Scatter(x=x, y=yt[idx], name="actual", line=dict(color=INK, width=1.6), showlegend=(r == 1)), row=r, col=1)
        miss = (yt[idx] < lo) | (yt[idx] > hi)
        fig.add_trace(go.Scatter(x=x[miss], y=yt[idx][miss], mode="markers", name="outside the interval", marker=dict(color=c, size=7, line=dict(color="white", width=1)), showlegend=False), row=r, col=1)
        fig.layout.annotations[r - 1].text += f": {100 * (1 - miss.mean()):.0f}% of this stretch covered"
    for tr in fig.data:
        if tr.name == "90% interval": tr.showlegend = False
    fig.update_layout(**LAYOUT, title=f"{title}: {H}-step-ahead 90% intervals, last {len(idx)} {'hours' if freq == 'h' else 'days'} of the test period", width=1000, height=640)
    fig.update_yaxes(title_text=unit); fig.update_xaxes(title_text="hours" if freq == "h" else "days", row=2, col=1)
    fig.write_image(f"out/{series}_intervals.png", scale=2); fig.write_html(f"out/{series}_intervals.html", include_plotlyjs="cdn")

# 3. summary bars
fig = go.Figure()
ser = [NAMES[s][0] for s in ("births", "river", "power")]
for k, c in (("statsforecast default (2 windows)", RED), ("statsforecast 10 windows", AMBER), ("pyconformal multi-step split", BLUE)):
    v = [100 * r[5] for s in ser for r in rows if r[0] == s and r[4] == k]
    fig.add_trace(go.Bar(x=ser, y=v, name=k, marker_color=c, text=[f"{a:.0f}%" for a in v], textposition="outside"))
fig.add_hline(y=90, line=dict(color=INK, dash="dash", width=1.5), annotation_text="stated level: 90%", annotation_position="top left")
fig.update_layout(**LAYOUT, barmode="group", title="Coverage of 90% forecast intervals on three real series", yaxis=dict(title="coverage (%)", range=[0, 105]), width=1000, height=520)
fig.write_image("out/summary_coverage.png", scale=2); fig.write_html("out/summary_coverage.html", include_plotlyjs="cdn")
extra = []
for series, model, mname in [("births", "ets", "AutoETS"), ("river", "ets", "AutoETS"), ("power", "theta", "AutoTheta")]:
    yt, F, res, ok, out, H = summarise(series, model, verbose=False)
    for k in ("statsforecast default (2 windows)", "statsforecast 10 windows", "pyconformal multi-step split"):
        d = out[k]; extra.append((NAMES[series][0], mname, H, int(ok[:, 0].sum()), k, d["cov"], d["covh"].min(), d["width"]))
import pickle; pickle.dump((rows, extra), open("rows.pkl", "wb"))
# one page with every interactive figure
parts = []
for name in ("summary_coverage", "births_intervals", "births_coverage", "river_intervals", "river_coverage", "power_intervals", "power_coverage"):
    parts.append(open(f"out/{name}.html").read().split("<body>")[1].split("</body>")[0])
open("out/case_studies_interactive.html", "w").write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>pyconformal case studies</title>"
    "<style>body{font-family:Helvetica,Arial,sans-serif;max-width:1040px;margin:30px auto;color:#22262B}</style></head><body>"
    "<h1>pyconformal on real series</h1><p>90% forecast intervals from statsforecast 2.1.1 and from pyconformal 0.1.0 on the same point forecasts. Hover for values; drag to zoom.</p>"
    + "".join(parts) + "</body></html>")
for r in rows: print(f"{r[0]:36s} {r[4]:42s} {100*r[5]:.1f} {100*r[6]:.1f} {r[7]:.1f} n={r[3]}")
