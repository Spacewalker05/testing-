import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="I-V Curve Plotter", layout="wide")
st.title("I-V Curve Plotter")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Settings")
n = st.sidebar.number_input("Number of samples (n)", min_value=1, max_value=20, value=3, step=1)
mode = st.sidebar.radio("Data source", ["Demo data", "Manual entry", "Upload CSV"])
log_y = st.sidebar.checkbox("Log scale (|Current|)", value=False)
show_markers = st.sidebar.checkbox("Show data points", value=True)
v_unit = st.sidebar.text_input("Voltage unit", "V")
i_unit = st.sidebar.text_input("Current unit", "A")


# ---------------- Data helpers ----------------
def demo_data(n):
    """Diode-like I-V curves, one per sample."""
    v = np.linspace(0, 0.8, 40)
    samples = {}
    for k in range(n):
        i0 = 1e-12 * (k + 1)          # saturation current differs per sample
        ideality = 1.5 + 0.1 * k
        vt = 0.02585
        i = i0 * (np.exp(v / (ideality * vt)) - 1)
        samples[f"Sample {k + 1}"] = pd.DataFrame({"Voltage": v, "Current": i})
    return samples


def manual_data(n):
    """One editable table per sample."""
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
            )
    return samples


def csv_data(n):
    """First column = voltage, remaining columns = current for each sample."""
    st.info("CSV format: first column is Voltage, each following column is the Current of one sample.")
    file = st.file_uploader("Upload CSV", type="csv")
    samples = {}
    if file is not None:
        df = pd.read_csv(file)
        v_col = df.columns[0]
        for col in df.columns[1:n + 1]:
            samples[str(col)] = pd.DataFrame({"Voltage": df[v_col], "Current": df[col]})
    return samples


# ---------------- Load data ----------------
if mode == "Demo data":
    samples = demo_data(n)
elif mode == "Manual entry":
    samples = manual_data(n)
else:
    samples = csv_data(n)

# ---------------- Plot ----------------
if samples:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, df in samples.items():
        df = df.dropna().sort_values("Voltage")
        if df.empty:
            continue
        y = df["Current"].abs() if log_y else df["Current"]
        ax.plot(df["Voltage"], y, marker="o" if show_markers else None, markersize=4, label=name)

    ax.set_xlabel(f"Voltage ({v_unit})")
    ax.set_ylabel(f"Current ({i_unit})")
    ax.set_title("I-V Characteristics")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    # Download plot
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download plot (PNG)", buf.getvalue(), "iv_curve.png", "image/png")

    with st.expander("View data"):
        for name, df in samples.items():
            st.write(name)
            st.dataframe(df, use_container_width=True)
else:
    st.warning("No data to plot yet.")
