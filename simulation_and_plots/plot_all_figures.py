import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Typography
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10.5,
    "axes.titlesize": 11,
    "legend.fontsize": 8.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.35,
    "savefig.dpi": 300,
})


def save_dual_format(fig, assets_dir, pdf_dir, filename_base):
    png_path = os.path.join(assets_dir, f"{filename_base}.png")
    pdf_path = os.path.join(pdf_dir, f"{filename_base}.pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {filename_base}.png & .pdf")


def generate_all_plots():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(repo_root, "data", "indoor_evac_benchmark.csv")
    assets_dir = os.path.join(repo_root, "assets")
    pdf_dir = os.path.join(assets_dir, "overleaf_pdfs")

    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Run generate_dataset_and_run.py first."
        )

    df = pd.read_csv(csv_path)
    t = df["time_s"].values

    # =========================================================================
    # Figure 1: Multi-Floor Stair Descent Altitude Tracking Comparison
    # =========================================================================
    fig1, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axvspan(42, 74, color="#ffcccc", alpha=0.35, label="Thermal Plume Zone (24°C → 47.5°C)")

    ax.plot(t, df["raw_baro_alt_m"], color="#9aa0a6", linewidth=1.0, alpha=0.85, label="Raw BMP280 Barometer")
    ax.plot(t, df["std_ekf_alt_m"], color="#e67e22", linestyle="-.", linewidth=1.35, label="Standard EKF (Fixed R)")
    ax.plot(t, df["true_alt_m"], color="#111111", linestyle="--", linewidth=1.6, label="Ground Truth Trajectory")
    ax.plot(t, df["cnn_ekf_alt_m"], color="#0b5394", linewidth=1.8, label="Proposed 1D-CNN + EKF")

    ax.set_title("Multi-Floor Vertical Altitude Tracking Under Thermal Drift & Pressure Transients")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Altitude (m)")
    ax.set_xlim(0, 120)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    save_dual_format(fig1, assets_dir, pdf_dir, "fig1_altitude_tracking")

    # =========================================================================
    # Figure 2: Thermal Spike & 1D-CNN Dynamic Covariance Multiplier Response
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)

    # Top subplot: Ambient temperature vs. barometric error
    color_temp = "#c0392b"
    ax1.plot(t, df["temp_c"], color=color_temp, linewidth=1.6, label="Ambient Temperature (°C)")
    ax1.set_ylabel("Temperature (°C)", color=color_temp)
    ax1.tick_params(axis="y", labelcolor=color_temp)
    ax1.set_title("1D-CNN Anomaly Detection & Adaptive Covariance Scaling")

    ax1_twin = ax1.twinx()
    ax1_twin.grid(False)
    baro_err = np.abs(df["raw_baro_alt_m"] - df["true_alt_m"])
    ax1_twin.plot(t, baro_err, color="#7f8c8d", linestyle=":", linewidth=1.2, label="|Raw Baro Error| (m)")
    ax1_twin.set_ylabel("Baro Drift Error (m)", color="#555555")

    # Bottom subplot: CNN noise scaling factor alpha_k
    ax2.plot(t, df["cnn_alpha_scale"], color="#1f618d", linewidth=1.6, label="1D-CNN Covariance Multiplier ($\\alpha_k = R_k / R_{base}$)")
    ax2.fill_between(t, 1.0, df["cnn_alpha_scale"], color="#5dade2", alpha=0.25)
    ax2.set_ylabel("Noise Scale $\\alpha_k$")
    ax2.set_xlabel("Time (s)")
    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 26)
    ax2.legend(loc="upper right", frameon=True)

    fig2.tight_layout()
    save_dual_format(fig2, assets_dir, pdf_dir, "fig2_thermal_drift_rk")

    # =========================================================================
    # Figure 3: Error Distribution Boxplot & RMSE Comparison Across Methods
    # =========================================================================
    fig3, ax = plt.subplots(figsize=(6.8, 3.6))
    true_z = df["true_alt_m"].values
    errors = [
        np.abs(df["raw_baro_alt_m"].values - true_z),
        np.abs(df["comp_filter_alt_m"].values - true_z),
        np.abs(df["std_ekf_alt_m"].values - true_z),
        np.abs(df["cnn_ekf_alt_m"].values - true_z),
    ]
    labels = ["Raw BMP280", "Complementary\nFilter", "Standard EKF\n(Fixed R)", "Proposed\n1D-CNN + EKF"]
    colors = ["#bdc3c7", "#f5b041", "#eb984e", "#2e86c1"]

    bp = ax.boxplot(errors, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.45)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.5)

    # Annotate exact RMSE above each box
    for idx, err_arr in enumerate(errors, start=1):
        rmse = np.sqrt(np.mean(err_arr ** 2))
        upper_whisker = np.percentile(err_arr, 93)
        ax.text(idx, upper_whisker + 0.12, f"RMSE:\n{rmse:.2f} m", ha="center", va="bottom", fontsize=8.8, fontweight="bold")

    ax.set_title("Vertical Positioning Absolute Error Distribution & RMSE")
    ax.set_ylabel("Absolute Altitude Error |e| (m)")
    ax.set_ylim(-0.05, max(np.percentile(errors[0], 95) + 0.65, 2.8))
    save_dual_format(fig3, assets_dir, pdf_dir, "fig3_error_boxplot")

    # =========================================================================
    # Figure 4: Decentralized Ad-Hoc Mesh Multi-Hop Latency & Delivery Rate
    # =========================================================================
    hops = np.array([1, 2, 3, 4, 5])
    latency_ms = np.array([14.2, 31.8, 52.4, 71.9, 93.5])
    latency_std = np.array([2.1, 4.5, 6.8, 9.4, 12.1])
    pdr_pct = np.array([99.6, 98.8, 97.4, 95.6, 93.2])

    fig4, ax1 = plt.subplots(figsize=(6.8, 3.5))
    bar_color = "#5499c7"
    bars = ax1.bar(hops - 0.16, latency_ms, yerr=latency_std, width=0.32, color=bar_color,
                   capsize=4, alpha=0.9, label="End-to-End Latency (ms)")
    ax1.set_xlabel("Mesh Relay Hop Count")
    ax1.set_ylabel("End-to-End Latency (ms)", color="#1f618d")
    ax1.tick_params(axis="y", labelcolor="#1f618d")
    ax1.set_xticks(hops)
    ax1.set_xticklabels([f"{h} Hop" if h == 1 else f"{h} Hops" for h in hops])
    ax1.set_ylim(0, 125)

    ax2 = ax1.twinx()
    ax2.grid(False)
    line_color = "#1e8449"
    ax2.plot(hops + 0.16, pdr_pct, color=line_color, marker="o", linewidth=1.8, markersize=6,
             label="Packet Delivery Ratio (%)")
    for h, p in zip(hops, pdr_pct):
        ax2.text(h + 0.16, p + 0.6, f"{p:.1f}%", ha="center", va="bottom", fontsize=8.5, color=line_color, fontweight="bold")
    ax2.set_ylabel("Packet Delivery Ratio (%)", color=line_color)
    ax2.tick_params(axis="y", labelcolor=line_color)
    ax2.set_ylim(85, 102)

    ax1.set_title("painlessMesh Multi-Hop Broadcast Performance (24-Byte Packed Frame)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)
    save_dual_format(fig4, assets_dir, pdf_dir, "fig4_mesh_performance")


if __name__ == "__main__":
    generate_all_plots()
