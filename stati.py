import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.ticker import AutoMinorLocator
import streamlit as st

st.set_page_config(page_title="Statistical Scatter Plot", layout="wide")
st.title("Statistical Scatter Plot (FF, PCE, Voc, Jsc ...)")

# ===== FUNCTIONS =====
PRETTY = {
    "pce": "PCE (%)",
    "ff": "FF (%)",
    "voc": "$V_{oc}$ (V)",
    "jsc": "$J_{sc}$ (mA cm$^{-2}$)",
}
ORIGIN_COLORS = ["#000000", "#FF0000", "#0000FF", "#00A000", "#FF00FF",
                 "#FF8000", "#00B0F0", "#7030A0", "#8B4513", "#808080"]


def pretty(name: str) -> str:
    return PRETTY.get(str(name).strip().lower(), str(name))


def _read(raw: str) -> pd.DataFrame:
    raw = raw.strip()
    df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", header=None, dtype=str)
    if df.shape[1] < 2:  # delimiter not detected -> split on whitespace
        df = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python", header=None, dtype=str)
    return df


def _unique(names):
    seen, out = {}, []
    for k, nm in enumerate(names, start=1):
        nm = "" if pd.isna(nm) else str(nm).strip()
        nm = nm or f"Group {k}"
        if nm in seen:
            seen[nm] += 1
            nm = f"{nm} ({seen[nm]})"
        else:
            seen[nm] = 1
        out.append(nm)
    return out


def parse_wide(raw: str) -> dict:
    """Each column = one group (one parameter). Header row optional."""
    if not raw.strip():
        return {}
    df = _read(raw)
    has_header = pd.to_numeric(df.iloc[0], errors="coerce").isna().any()
    if has_header:
        names = _unique(df.iloc[0].tolist())
        df = df.iloc[1:]
    else:
        names = [f"Group {k}" for k in range(1, df.shape[1] + 1)]
    groups = {}
    for k, nm in enumerate(names):
        vals = pd.to_numeric(df.iloc[:, k], errors="coerce").dropna().to_numpy(float)
        if len(vals):
            groups[nm] = vals
    if not groups:
        raise ValueError("No numeric data found.")
    return groups


def parse_long(raw: str) -> pd.DataFrame:
    """First column = group name, other columns = parameters (FF, PCE, Voc, Jsc ...). Header required."""
    df = _read(raw)
    names = _unique(df.iloc[0].tolist())
    df = df.iloc[1:].reset_index(drop=True)
    df.columns = names
    df[names[0]] = df[names[0]].ffill().astype(str).str.strip()
    for c in names[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=names[1:], how="all")
    if df.empty or len(names) < 2:
        raise ValueError("Need a header row: Group, then one column per parameter.")
    return df


def stats_table(groups: dict) -> pd.DataFrame:
    rows = []
    for nm, v in groups.items():
        sd = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append({
            "Group": nm, "n": len(v), "Mean": v.mean(), "SD": sd,
            "SEM": sd / np.sqrt(len(v)) if len(v) > 1 else np.nan,
            "Median": np.median(v), "Min": v.min(), "Max": v.max(),
        })
    return pd.DataFrame(rows)


def make_plot(groups, ylabel, colors, o):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams["mathtext.default"] = "regular"
    fs = o["font"]
    rng = np.random.default_rng(o["seed"])
    n = len(groups)
    w = min(max(0.95 * n + 1.8, 3.4), 12) if o["auto_w"] else o["w"]

    fig, ax = plt.subplots(figsize=(w, o["h"]))
    lows, highs = [], []
    stats = {}
    for nm, vals in groups.items():
        m = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        e = {"SD": sd, "SEM": sd / np.sqrt(len(vals)), "None": 0.0}[o["err"]]
        stats[nm] = (m, e)
        if o["style"] == "box":
            lows.append(vals.min()); highs.append(vals.max())
        else:
            lows.append(min(vals.min(), m - e)); highs.append(max(vals.max(), m + e))
    span = (max(highs) - min(lows)) or 1.0

    for i, (nm, vals) in enumerate(groups.items()):
        x, c = i + 1, colors[nm]
        m, e = stats[nm]

        if o["style"] == "box":
            bp = ax.boxplot(vals, positions=[x], widths=0.55, showfliers=False,
                            patch_artist=True, manage_ticks=False)
            for patch in bp["boxes"]:
                patch.set_facecolor(to_rgba(c, 0.18))
                patch.set_edgecolor(c)
                patch.set_linewidth(1.4)
            for key in ("whiskers", "caps"):
                plt.setp(bp[key], color=c, linewidth=1.4)
            plt.setp(bp["medians"], color=c, linewidth=2.0)

        jitter = rng.uniform(-o["jitter"], o["jitter"], len(vals))
        ax.scatter(x + jitter, vals, s=o["size"], marker=o["marker"], facecolor=c,
                   edgecolor="white", linewidth=0.8, alpha=o["alpha"], zorder=3)

        if o["style"] == "box":
            ax.scatter(x, m, marker="D", s=o["size"] * 0.8, facecolor="white",
                       edgecolor="k", linewidth=1.2, zorder=5)
        else:
            ax.hlines(m, x - 0.28, x + 0.28, colors="k", linewidth=2.0, zorder=4)
            if e > 0:
                ax.errorbar(x, m, yerr=e, fmt="none", ecolor="k", elinewidth=1.3,
                            capsize=4, capthick=1.3, zorder=4)

        if o["show_vals"]:
            top = vals.max() if o["style"] == "box" else max(vals.max(), m + e)
            ax.text(x, top + 0.04 * span, f"{m:.{o['dec']}f}", ha="center", va="bottom",
                    fontsize=fs - 2, color="k")

    ax.set_xlim(0.5, n + 0.5)
    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(list(groups), rotation=o["rot"],
                       ha="right" if o["rot"] not in (0, 90) else "center", fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs + 1)
    if o["ylim"] is not None:
        ax.set_ylim(*o["ylim"])
    else:
        ax.set_ylim(min(lows) - 0.12 * span, max(highs) + (0.22 if o["show_vals"] else 0.12) * span)

    for sp in ax.spines.values():
        sp.set_linewidth(1.2)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", direction="in", width=1.2, length=5,
                   labelsize=fs, top=False, right=True, pad=5)
    ax.tick_params(axis="y", which="minor", direction="in", width=1.0, length=2.5, right=True)
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
    return buf.getvalue()


# ===== UI =====
st.sidebar.header("Input")
fmt = st.sidebar.radio(
    "Input format",
    ["One parameter (each column = one group)", "All parameters (Group, FF, PCE, Voc, Jsc ...)"],
)

if fmt.startswith("One"):
    param = st.sidebar.selectbox("Parameter", ["PCE", "FF", "Voc", "Jsc", "Custom"])
    default_label = pretty(param) if param != "Custom" else "Value"
else:
    param = None
    default_label = None

EX_WIDE = ("Control\tTreated\tOptimized\n"
           "17.8\t19.1\t20.4\n17.2\t19.6\t21.0\n18.1\t18.7\t20.1\n17.5\t19.3\t20.8\n"
           "16.9\t18.9\t20.6\n18.3\t19.8\t19.9\n17.6\t19.0\t20.3\n17.9\t19.4\t20.9")
EX_LONG = ("Group\tPCE\tFF\tVoc\tJsc\n"
           "Control\t17.8\t74.1\t1.06\t22.7\nControl\t17.2\t73.0\t1.05\t22.4\nControl\t18.1\t75.2\t1.07\t22.5\n"
           "Treated\t19.1\t77.5\t1.10\t22.8\nTreated\t19.6\t78.3\t1.11\t22.6\nTreated\t18.7\t76.9\t1.09\t22.9\n"
           "Optimized\t20.4\t80.1\t1.13\t22.5\nOptimized\t21.0\t81.0\t1.14\t22.7\nOptimized\t20.1\t79.5\t1.12\t22.6")

groups, ylabel_auto = {}, "Value"
try:
    if fmt.startswith("One"):
        st.write("Paste the values of **one parameter**. Each column is one group (e.g. Control, "
                 "Treated, ...) and each row is one device. A header row with group names is optional.")
        raw = st.text_area("Paste data here", value=EX_WIDE, height=220)
        groups = parse_wide(raw)
        ylabel_auto = default_label
    else:
        st.write("Paste a table with a header row: first column = group name, other columns = "
                 "parameters. Then pick the parameter to plot.")
        raw = st.text_area("Paste data here", value=EX_LONG, height=220)
        table = parse_long(raw)
        gcol, pcols = table.columns[0], list(table.columns[1:])
        pick = st.selectbox("Parameter to plot", pcols)
        for nm, sub in table.groupby(gcol, sort=False):
            v = sub[pick].dropna().to_numpy(float)
            if len(v):
                groups[nm] = v
        ylabel_auto = pretty(pick)
except Exception as e:
    st.error(f"Could not read the data: {e}")

# ---------------- Plot settings ----------------
st.sidebar.header("Plot style")
ylabel = st.sidebar.text_input("Y-axis label (blank = automatic)", "") or ylabel_auto
style_name = st.sidebar.selectbox("Plot style", ["Scatter + mean ± error", "Box + scatter"])
style = "box" if style_name.startswith("Box") else "scatter"
err = st.sidebar.radio("Error bars (scatter style)", ["SD", "SEM", "None"], horizontal=True)
show_vals = st.sidebar.checkbox("Show mean value above each group", False)
dec = st.sidebar.slider("Decimals for mean value", 0, 4, 2) if show_vals else 2
size = st.sidebar.slider("Point size", 10, 200, 45, 5)
jitter = st.sidebar.slider("Jitter width", 0.0, 0.4, 0.14, 0.01)
alpha = st.sidebar.slider("Point opacity", 0.2, 1.0, 0.9, 0.05)
marker = st.sidebar.selectbox("Marker", ["o", "s", "^", "D", "v"])
font = st.sidebar.slider("Font size", 8, 24, 13)
auto_w = st.sidebar.checkbox("Auto figure width (fits number of groups)", True)
w = st.sidebar.slider("Figure width (in)", 3.0, 12.0, 5.0, 0.5, disabled=auto_w)
h = st.sidebar.slider("Figure height (in)", 3.0, 10.0, 4.2, 0.5)
rot = st.sidebar.selectbox("X label rotation", [0, 30, 45, 90])
seed = st.sidebar.number_input("Jitter seed", 0, 9999, 1)

ylim = None
if groups and st.sidebar.checkbox("Set Y-axis range manually"):
    allv = np.concatenate(list(groups.values()))
    pad = (allv.max() - allv.min()) * 0.1 or 1.0
    lo = st.sidebar.number_input("Y min", value=float(allv.min() - pad), format="%.4f")
    hi = st.sidebar.number_input("Y max", value=float(allv.max() + pad), format="%.4f")
    ylim = (lo, hi)

# ---------------- Output ----------------
if groups:
    colors = {}
    with st.sidebar.expander("Group colours"):
        for k, nm in enumerate(groups):
            colors[nm] = st.color_picker(nm, ORIGIN_COLORS[k % len(ORIGIN_COLORS)], key=f"col_{k}")

    opts = dict(style=style, err=err, show_vals=show_vals, dec=dec, size=size, jitter=jitter,
                alpha=alpha, marker=marker, font=font, auto_w=auto_w, w=w, h=h, rot=rot,
                seed=int(seed), ylim=ylim)
    fig = make_plot(groups, ylabel, colors, opts)

    left, _ = st.columns([2, 1])
    with left:
        st.pyplot(fig, bbox_inches="tight")

    c1, c2, c3 = st.columns(3)
    c1.download_button("Download PNG (300 dpi)", fig_bytes(fig, "png"), "scatter.png", "image/png")
    c2.download_button("Download SVG", fig_bytes(fig, "svg"), "scatter.svg", "image/svg+xml")
    c3.download_button("Download PDF", fig_bytes(fig, "pdf"), "scatter.pdf", "application/pdf")
    plt.close(fig)

    st.subheader("Statistics")
    st.dataframe(stats_table(groups).round(4), use_container_width=True, hide_index=True)
