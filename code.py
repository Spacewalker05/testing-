import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="I-V Curve Plotter", layout="wide")
st.title("I-V Curve Plotter")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Data source", ["Paste data (copy/paste)", "Upload CSV", "Manual entry", "Demo data"])
n = None
if mode in ("Manual entry", "Demo data"):
    n = st.sidebar.number_input("Number of samples (n)", min_value=1, max_value=20, value=3, step=1)
log_y = st.sidebar.checkbox("Log scale (|Current|)", value=False)
show_markers = st.sidebar.checkbox("Show data points", value=True)
sort_v = st.sidebar.checkbox("Sort by voltage (turn OFF for forward/reverse sweeps)", value=False)
layout = st.sidebar.selectbox(
    "Column layout",
    ["Auto", "Pairs: V, I, V, I, V, I ...", "Shared voltage: V, I1, I2, I3 ..."],
)
v_unit = st.sidebar.text_input("Voltage unit", "V")
i_unit = st.sidebar.text_input("Current unit", "A")


# ---------------- Parsing helpers ----------------
def parse_table(raw: str, layout: str = "Auto") -> dict:
    """Turn pasted/uploaded text into {sample_name: DataFrame(Voltage, Current)}.

    Layouts:
      Pairs  : V, I, V, I, V, I ...   (every sample has its own voltage column)
      Shared : V, I1, I2, I3 ...      (one voltage column shared by all samples)
      Auto   : guesses from the header names / column count
    Separator (comma, tab, semicolon, space) and header row are detected automatically.
    """
    raw = raw.strip()
    if not raw:
        return {}

    df = pd.read_csv(io.StringIO(raw), sep=None, engine="python", header=None)
    if df.shape[1] < 2:  # delimiter not detected -> split on any whitespace
        df = pd.read_csv(io.StringIO(raw), sep=r"\s+", engine="python", header=None)
    if df.shape[1] < 2:
        raise ValueError("Need at least 2 columns: Voltage and Current.")

    # Detect header row (first row not fully numeric)
    has_header = pd.to_numeric(df.iloc[0], errors="coerce").isna().any()
    if has_header:
        names = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        names = [""] * df.shape[1]

    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if df.empty:
        raise ValueError("No numeric data found.")
    ncol = df.shape[1]

    # Decide layout
    if layout.startswith("Pairs"):
        pairs = True
    elif layout.startswith("Shared"):
        pairs = False
    elif has_header:
        n_volt = sum(1 for nm in names if re.search(r"volt|^v($|[\s_(\[])", nm, re.I))
        pairs = n_volt > 1
    else:
        pairs = ncol >= 4 and ncol % 2 == 0

    samples = {}
    if pairs:
        idx = list(range(0, ncol - 1, 2))
        cur_names = [names[k + 1] for k in idx]
        unique = has_header and all(cur_names) and len(set(cur_names)) == len(cur_names)
        for n_, k in enumerate(idx, start=1):
            sub = pd.DataFrame({"Voltage": df.iloc[:, k], "Current": df.iloc[:, k + 1]}).dropna()
            if not sub.empty:
                samples[cur_names[n_ - 1] if unique else f"Sample {n_}"] = sub
    else:
        cur_names = names[1:]
        unique = has_header and all(cur_names) and len(set(cur_names)) == len(cur_names)
        for n_, k in enumerate(range(1, ncol), start=1):
            sub = pd.DataFrame({"Voltage": df.iloc[:, 0], "Current": df.iloc[:, k]}).dropna()
            if not sub.empty:
                samples[cur_names[n_ - 1] if unique else f"Sample {n_}"] = sub
    return samples


def demo_data(n):
    v = np.linspace(0, 0.8, 40)
    samples = {}
    for k in range(n):
        i0 = 1e-12 * (k + 1)
        ideality = 1.5 + 0.1 * k
        i = i0 * (np.exp(v / (ideality * 0.02585)) - 1)
        samples[f"Sample {k + 1}"] = pd.DataFrame({"Voltage": v, "Current": i})
    return samples


def manual_data(n):
    samples = {}
    tabs = st.tabs([f"Sample {k + 1}" for k in range(n)])
    for k, tab in enumerate(tabs):
        with tab:
            default = pd.DataFrame({
                "Voltage": [0.0, 0.2, 0.4, 0.6, 0.8],
                "Current": [0.0, 0.001, 0.004, 0.02, 0.1],
            })
            samples[f"Sample {k + 1}"] = st.data_editor(
                default, num_rows="dynamic", key=f"editor_{k}", use_container_width=True
            ).dropna()
    return samples


# ---------------- Load data ----------------
samples = {}
try:
    if mode == "Paste data (copy/paste)":
        st.write("Paste your data below (e.g. copied from Excel). Use any number of samples, either as "
                 "**Voltage, Current, Voltage, Current, ...** pairs or as one Voltage column followed by "
                 "several Current columns. A header row is optional.")
        example = "Voltage\tSample A\tSample B\n0.0\t0.000\t0.000\n0.2\t0.001\t0.002\n0.4\t0.004\t0.006\n0.6\t0.020\t0.030\n0.8\t0.100\t0.150"
        text = st.text_area("Paste data here", value=example, height=250)
        samples = parse_table(text, layout)

    elif mode == "Upload CSV":
        st.info("Pairs (V, I, V, I, ...) or one shared Voltage column followed by Current columns.")
        file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
        if file is not None:
            samples = parse_table(file.getvalue().decode("utf-8-sig", errors="ignore"), layout)

    elif mode == "Manual entry":
        samples = manual_data(n)

    else:
        samples = demo_data(n)
except Exception as e:
    st.error(f"Could not read the data: {e}")

# ---------------- Plot ----------------
if samples:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in samples.items():
        if sort_v:
            df = df.sort_values("Voltage")
        y = df["Current"].abs() if log_y else df["Current"]
        ax.plot(df["Voltage"], y, marker="o" if show_markers else None, markersize=3, linewidth=1.2, label=name)

    ax.set_xlabel(f"Voltage ({v_unit})")
    ax.set_ylabel(f"Current ({i_unit})")
    ax.set_title("I-V Characteristics")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    st.download_button("Download plot (PNG)", buf.getvalue(), "iv_curve.png", "image/png")

    with st.expander("View parsed data"):
        for name, df in samples.items():
            st.write(name)
            st.dataframe(df, use_container_width=True)
elif mode == "Upload CSV":
    st.warning("Upload a file to see the plot.")
