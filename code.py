import io

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
v_unit = st.sidebar.text_input("Voltage unit", "V")
i_unit = st.sidebar.text_input("Current unit", "A")


# ---------------- Parsing helpers ----------------
def parse_table(raw: str) -> dict:
    """Turn pasted/uploaded text into {sample_name: DataFrame(Voltage, Current)}.

    First column = Voltage, every other column = Current of one sample.
    Works with comma, tab, semicolon or space separated data, with or without a header row.
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
    first_row = pd.to_numeric(df.iloc[0], errors="coerce")
    if first_row.isna().any():
        names = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        names = ["Voltage"] + [f"Sample {k}" for k in range(1, df.shape[1])]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    if df.empty:
        raise ValueError("No numeric data found.")

    samples = {}
    for k in range(1, df.shape[1]):
        sub = pd.DataFrame({"Voltage": df.iloc[:, 0], "Current": df.iloc[:, k]}).dropna()
        if not sub.empty:
            samples[names[k]] = sub
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
        st.write("Paste your data below (e.g. copied from Excel). First column = Voltage, "
                 "other columns = Current of each sample. A header row is optional.")
        example = "Voltage\tSample A\tSample B\n0.0\t0.000\t0.000\n0.2\t0.001\t0.002\n0.4\t0.004\t0.006\n0.6\t0.020\t0.030\n0.8\t0.100\t0.150"
        text = st.text_area("Paste data here", value=example, height=250)
        samples = parse_table(text)

    elif mode == "Upload CSV":
        st.info("First column = Voltage, every other column = Current of one sample.")
        file = st.file_uploader("Upload CSV or TXT", type=["csv", "txt"])
        if file is not None:
            samples = parse_table(file.getvalue().decode("utf-8-sig", errors="ignore"))

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
