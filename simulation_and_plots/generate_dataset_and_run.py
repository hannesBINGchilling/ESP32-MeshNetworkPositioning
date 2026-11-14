import os
import numpy as np
import pandas as pd


class CnnEkfSimulator:
    """
    Exact Python mirror of the embedded C++ CnnEkfFilter (firmware/src/cnn_ekf_filter.cpp)
    for reproducible benchmarking and dataset generation.
    """

    def __init__(self, dt=0.02):
        self.dt = dt
        self.window_size = 16
        self.q_alt = 0.01
        self.q_vel = 0.08
        self.r_base = 0.35
        self.sea_level_hpa = 1013.25

        # Exact pre-trained weights matching firmware/src/cnn_ekf_filter.cpp
        self.conv1_w = np.array([
            [[ 0.42,  0.35,  0.10], [ 0.55,  0.48,  0.15], [ 0.40,  0.32,  0.10]],
            [[-0.45, -0.38,  0.12], [-0.58, -0.50,  0.18], [-0.42, -0.35,  0.12]],
            [[ 0.08,  0.15,  0.65], [ 0.10,  0.20,  0.80], [ 0.08,  0.15,  0.65]],
            [[ 0.50, -0.20,  0.05], [-0.85,  0.45,  0.05], [ 0.50, -0.20,  0.05]],
        ], dtype=np.float32)
        self.conv1_b = np.array([-0.08, -0.08, -0.10, -0.12], dtype=np.float32)

        self.conv2_w = np.array([
            [[0.35, 0.10, 0.15, 0.10], [0.45, 0.12, 0.20, 0.15], [0.35, 0.10, 0.15, 0.10]],
            [[0.10, 0.38, 0.18, 0.10], [0.12, 0.48, 0.22, 0.15], [0.10, 0.38, 0.18, 0.10]],
            [[0.15, 0.15, 0.45, 0.05], [0.20, 0.20, 0.55, 0.08], [0.15, 0.15, 0.45, 0.05]],
            [[0.12, 0.12, 0.08, 0.42], [0.15, 0.15, 0.10, 0.52], [0.12, 0.12, 0.08, 0.42]],
        ], dtype=np.float32)
        self.conv2_b = np.array([-0.05, -0.05, -0.05, -0.05], dtype=np.float32)

        self.dense_w = np.array([1.15, 1.20, 1.35, 0.95], dtype=np.float32)
        self.dense_b = -0.15

        self.window = np.zeros((self.window_size, 3), dtype=np.float32)
        self.x = np.zeros(3, dtype=np.float64)
        self.P = np.diag([0.25, 0.10, 0.02])

        self.ref_alt = 0.0
        self.prev_baro_alt = 0.0
        self.filtered_temp = 24.0
        self.temp_rate = 0.0
        self.initialized = False

    def pressure_to_alt(self, pressure_hpa):
        return 44330.0 * (1.0 - (pressure_hpa / self.sea_level_hpa) ** 0.19029495)

    def run_cnn(self):
        # Conv1D Layer 1 (valid padding: 16 -> 14)
        l1 = np.zeros((14, 4), dtype=np.float32)
        for t in range(14):
            patch = self.window[t:t + 3]  # (3, 3)
            l1[t] = np.maximum(0.0, np.tensordot(self.conv1_w, patch, axes=([1, 2], [0, 1])) + self.conv1_b)

        # Conv1D Layer 2 (valid padding: 14 -> 12)
        l2 = np.zeros((12, 4), dtype=np.float32)
        for t in range(12):
            patch = l1[t:t + 3]  # (3, 4)
            l2[t] = np.maximum(0.0, np.tensordot(self.conv2_w, patch, axes=([1, 2], [0, 1])) + self.conv2_b)

        gap = np.mean(l2, axis=0)
        raw_score = float(np.dot(gap, self.dense_w) + self.dense_b)
        scaled = raw_score * 0.45 - 0.30
        norm_act = np.tanh(scaled) if scaled > 0.0 else 0.0
        return float(1.0 + 24.0 * norm_act)

    def step(self, pressure_hpa, temp_c, accel_z):
        if not self.initialized:
            self.ref_alt = self.pressure_to_alt(pressure_hpa)
            self.filtered_temp = temp_c
            self.initialized = True

        raw_baro_alt = self.pressure_to_alt(pressure_hpa) - self.ref_alt
        baro_delta = raw_baro_alt - self.prev_baro_alt
        self.prev_baro_alt = raw_baro_alt

        prev_temp = self.filtered_temp
        self.filtered_temp = 0.85 * self.filtered_temp + 0.15 * temp_c
        inst_rate = (self.filtered_temp - prev_temp) / self.dt
        self.temp_rate = 0.80 * self.temp_rate + 0.20 * inst_rate

        # EKF Prediction
        dt = self.dt
        u = accel_z - self.x[2]
        z_pred = self.x[0] + self.x[1] * dt + 0.5 * u * dt * dt
        v_pred = self.x[1] + u * dt
        b_pred = self.x[2]

        F = np.array([
            [1.0, dt, -0.5 * dt * dt],
            [0.0, 1.0, -dt],
            [0.0, 0.0, 1.0]
        ])
        Q = np.diag([self.q_alt * dt, self.q_vel * dt, 0.0005 * dt])
        P_pred = F @ self.P @ F.T + Q

        # Sliding window update + 1D-CNN inference
        innovation = raw_baro_alt - z_pred
        self.window[:-1] = self.window[1:]
        self.window[-1] = [baro_delta * 10.0, innovation, self.temp_rate]

        alpha = self.run_cnn()
        r_k = alpha * self.r_base

        # EKF Update
        S = P_pred[0, 0] + r_k
        K = P_pred[:, 0] / S
        self.x = np.array([z_pred, v_pred, b_pred]) + K * innovation
        self.P = P_pred - np.outer(K, P_pred[0, :])

        return self.x[0], self.x[1], raw_baro_alt, alpha, r_k


def run_standard_ekf(pressure_hpa, accel_z, dt=0.02):
    """Baseline 3-state EKF with fixed measurement covariance R."""
    sea_level_hpa = 1013.25
    alt_meas = 44330.0 * (1.0 - (pressure_hpa / sea_level_hpa) ** 0.19029495)
    alt_meas -= alt_meas[0]

    x = np.zeros(3)
    P = np.diag([0.25, 0.10, 0.02])
    Q = np.diag([0.01 * dt, 0.08 * dt, 0.0005 * dt])
    R = 0.35
    F = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])

    out = np.zeros(len(alt_meas))
    for i in range(len(alt_meas)):
        u = accel_z[i] - x[2]
        x_pred = np.array([x[0] + x[1] * dt + 0.5 * u * dt * dt, x[1] + u * dt, x[2]])
        P_pred = F @ P @ F.T + Q
        innov = alt_meas[i] - x_pred[0]
        S = P_pred[0, 0] + R
        K = P_pred[:, 0] / S
        x = x_pred + K * innov
        P = P_pred - np.outer(K, P_pred[0, :])
        out[i] = x[0]
    return out


def run_complementary_filter(pressure_hpa, accel_z, dt=0.02):
    """Baseline 2nd-order Complementary Filter."""
    sea_level_hpa = 1013.25
    alt_meas = 44330.0 * (1.0 - (pressure_hpa / sea_level_hpa) ** 0.19029495)
    alt_meas -= alt_meas[0]

    z = 0.0
    vz = 0.0
    out = np.zeros(len(alt_meas))
    kp, ki = 1.6, 0.45
    acc_bias = 0.0
    for i in range(len(alt_meas)):
        err = alt_meas[i] - z
        acc_bias += ki * err * dt
        vz += (accel_z[i] - acc_bias + kp * err) * dt
        z += vz * dt
        out[i] = z
    return out


def generate_benchmark():
    rng = np.random.default_rng(42)
    fs = 50.0
    dt = 1.0 / fs
    duration = 120.0
    t = np.arange(0, duration, dt)
    n = len(t)

    # 1. C2-smooth 4-floor stair descent trajectory (12.0m -> 0.0m, 3.0m per floor)
    true_alt = np.full(n, 12.0)
    true_vel = np.zeros(n)
    true_acc = np.zeros(n)

    descents = [
        (10.0, 24.0, -3.0),   # Floor 5 -> Floor 4
        (34.0, 48.0, -3.0),   # Floor 4 -> Floor 3
        (62.0, 76.0, -3.0),   # Floor 3 -> Floor 2
        (88.0, 102.0, -3.0),  # Floor 2 -> Ground (Floor 1)
    ]

    for t0, t1, dz in descents:
        T = t1 - t0
        mask = (t >= t0) & (t <= t1)
        tau = t[mask] - t0
        true_alt[mask] += dz * (tau / T - np.sin(2.0 * np.pi * tau / T) / (2.0 * np.pi))
        true_vel[mask] = (dz / T) * (1.0 - np.cos(2.0 * np.pi * tau / T))
        true_acc[mask] = (2.0 * np.pi * dz / (T * T)) * np.sin(2.0 * np.pi * tau / T)
        true_alt[t > t1] += dz

    # 2. Simulate MPU-6050 linear vertical acceleration (with walking step harmonics & bias)
    walking_mask = np.zeros(n)
    for t0, t1, _ in descents:
        walking_mask[(t >= t0) & (t <= t1)] = 1.0
    step_harmonics = walking_mask * 0.45 * np.sin(2.0 * np.pi * 1.85 * t)
    accel_bias = 0.035 + np.cumsum(rng.normal(0.0, 0.00015, n))
    accel_z_meas = true_acc + step_harmonics + accel_bias + rng.normal(0.0, 0.11, n)

    # 3. Simulate ambient temperature spike (fire/smoke plume zone between 42s and 74s)
    temp_c = np.full(n, 24.0)
    rise_mask = (t >= 42.0) & (t <= 60.0)
    temp_c[rise_mask] += 23.5 * 0.5 * (1.0 - np.cos(np.pi * (t[rise_mask] - 42.0) / 18.0))
    hold_mask = (t > 60.0) & (t <= 74.0)
    temp_c[hold_mask] += 23.5
    cool_mask = t > 74.0
    temp_c[cool_mask] += 23.5 * np.exp(-(t[cool_mask] - 74.0) / 22.0)
    temp_c += rng.normal(0.0, 0.05, n)

    # 4. Simulate BMP280 pressure with thermal drift + door-slam pressure transients
    thermal_drift_m = -0.105 * (temp_c - 24.0)  # Up to -2.47m false altitude drop under heat
    door_transient = (
        0.55 * np.exp(-((t - 28.0) / 0.6) ** 2)
        - 0.65 * np.exp(-((t - 83.0) / 0.5) ** 2)
    )
    baro_noise = rng.normal(0.0, 0.045, n)  # Hardware IIR filtered BMP280 noise
    apparent_alt = true_alt + thermal_drift_m + door_transient + baro_noise

    # Convert apparent altitude to hPa pressure (anchored at 1013.25 hPa at 0m)
    pressure_hpa = 1013.25 * (1.0 - apparent_alt / 44330.0) ** (1.0 / 0.19029495)

    # 5. Run all 4 estimation methods
    cnn_ekf = CnnEkfSimulator(dt=dt)
    est_cnn_alt = np.zeros(n)
    est_cnn_vel = np.zeros(n)
    raw_baro_rel = np.zeros(n)
    alpha_hist = np.zeros(n)
    rk_hist = np.zeros(n)

    for i in range(n):
        alt, vel, raw_b, alpha, rk = cnn_ekf.step(pressure_hpa[i], temp_c[i], accel_z_meas[i])
        est_cnn_alt[i] = alt
        est_cnn_vel[i] = vel
        raw_baro_rel[i] = raw_b
        alpha_hist[i] = alpha
        rk_hist[i] = rk

    # Align ground truth relative to initial barometric frame (starts at 0.0m relative)
    true_rel_alt = true_alt - true_alt[0]
    est_std_ekf = run_standard_ekf(pressure_hpa, accel_z_meas, dt=dt)
    est_comp = run_complementary_filter(pressure_hpa, accel_z_meas, dt=dt)

    # Save CSV benchmark dataset
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "indoor_evac_benchmark.csv")

    df = pd.DataFrame({
        "time_s": np.round(t, 3),
        "true_alt_m": np.round(true_rel_alt, 4),
        "true_vel_ms": np.round(true_vel, 4),
        "pressure_hpa": np.round(pressure_hpa, 4),
        "temp_c": np.round(temp_c, 3),
        "accel_z_ms2": np.round(accel_z_meas, 4),
        "raw_baro_alt_m": np.round(raw_baro_rel, 4),
        "comp_filter_alt_m": np.round(est_comp, 4),
        "std_ekf_alt_m": np.round(est_std_ekf, 4),
        "cnn_ekf_alt_m": np.round(est_cnn_alt, 4),
        "cnn_alpha_scale": np.round(alpha_hist, 3),
        "adaptive_rk": np.round(rk_hist, 4),
    })
    df.to_csv(csv_path, index=False)

    # Print quantitative benchmark metrics
    methods = {
        "Raw BMP280 Barometer": raw_baro_rel,
        "Complementary Filter": est_comp,
        "Standard EKF (Fixed R)": est_std_ekf,
        "Proposed 1D-CNN + EKF": est_cnn_alt,
    }

    print("\n" + "=" * 68)
    print(f"{'Method':<26} | {'RMSE (m)':>10} | {'MAE (m)':>10} | {'Max Err (m)':>11}")
    print("-" * 68)
    rmse_std = np.sqrt(np.mean((est_std_ekf - true_rel_alt) ** 2))
    rmse_cnn = np.sqrt(np.mean((est_cnn_alt - true_rel_alt) ** 2))
    for name, pred in methods.items():
        err = np.abs(pred - true_rel_alt)
        rmse = np.sqrt(np.mean(err ** 2))
        mae = np.mean(err)
        max_e = np.max(err)
        print(f"{name:<26} | {rmse:>10.3f} | {mae:>10.3f} | {max_e:>11.3f}")
    print("=" * 68)
    reduction = (1.0 - rmse_cnn / rmse_std) * 100.0
    print(f"[INFO] Saved dataset to: {csv_path}")
    print(f"[INFO] RMSE reduction vs. Standard EKF: {reduction:.1f}%\n")


if __name__ == "__main__":
    generate_benchmark()
