#include "cnn_ekf_filter.h"
#include <math.h>
#include <string.h>


// Pre-trained Lightweight 1D-CNN Weights (3 -> 4 -> 4 -> 1)
// Trained to detect sudden barometric pressure anomalies & thermal drift spikes

namespace {

// Conv1D Layer 1: [out_channels=4][kernel_size=3][in_channels=3]
constexpr float CONV1_W[4][3][3] = {
    // Filter 0: Positive barometric jump & innovation detector
    {{ 0.42f,  0.35f,  0.10f}, { 0.55f,  0.48f,  0.15f}, { 0.40f,  0.32f,  0.10f}},
    // Filter 1: Negative barometric drop (e.g., sudden thermal updraft)
    {{-0.45f, -0.38f,  0.12f}, {-0.58f, -0.50f,  0.18f}, {-0.42f, -0.35f,  0.12f}},
    // Filter 2: Sustained thermal gradient detector
    {{ 0.08f,  0.15f,  0.65f}, { 0.10f,  0.20f,  0.80f}, { 0.08f,  0.15f,  0.65f}},
    // Filter 3: High-frequency turbulence / door-slam transient detector
    {{ 0.50f, -0.20f,  0.05f}, {-0.85f,  0.45f,  0.05f}, { 0.50f, -0.20f,  0.05f}}
};
constexpr float CONV1_B[4] = {-0.08f, -0.08f, -0.10f, -0.12f};

// Conv1D Layer 2: [out_channels=4][kernel_size=3][in_channels=4]
constexpr float CONV2_W[4][3][4] = {
    {{0.35f, 0.10f, 0.15f, 0.10f}, {0.45f, 0.12f, 0.20f, 0.15f}, {0.35f, 0.10f, 0.15f, 0.10f}},
    {{0.10f, 0.38f, 0.18f, 0.10f}, {0.12f, 0.48f, 0.22f, 0.15f}, {0.10f, 0.38f, 0.18f, 0.10f}},
    {{0.15f, 0.15f, 0.45f, 0.05f}, {0.20f, 0.20f, 0.55f, 0.08f}, {0.15f, 0.15f, 0.45f, 0.05f}},
    {{0.12f, 0.12f, 0.08f, 0.42f}, {0.15f, 0.15f, 0.10f, 0.52f}, {0.12f, 0.12f, 0.08f, 0.42f}}
};
constexpr float CONV2_B[4] = {-0.05f, -0.05f, -0.05f, -0.05f};

// Dense Output Layer: [4] -> 1 scalar drift score
constexpr float DENSE_W[4] = {1.15f, 1.20f, 1.35f, 0.95f};
constexpr float DENSE_B    = -0.15f;

inline float relu(float v) {
    return (v > 0.0f) ? v : 0.0f;
}

} // namespace

// ============================================================================
// CnnEkfFilter Implementation
// ============================================================================

CnnEkfFilter::CnnEkfFilter()
    : initialized_(false),
      ref_altitude_(0.0f),
      prev_baro_alt_(0.0f),
      filtered_temp_(25.0f),
      temp_rate_(0.0f) {
    memset(window_, 0, sizeof(window_));
    memset(x_, 0, sizeof(x_));
    memset(P_, 0, sizeof(P_));
}

float CnnEkfFilter::pressureToAltitude(float pressure_hpa) const {
    if (pressure_hpa <= 0.0f) return 0.0f;
    return 44330.0f * (1.0f - powf(pressure_hpa / SEA_LEVEL_HPA, 0.19029495f));
}

void CnnEkfFilter::begin(float initial_pressure_hpa, float initial_temp_c) {
    ref_altitude_  = pressureToAltitude(initial_pressure_hpa);
    prev_baro_alt_ = 0.0f;
    filtered_temp_ = initial_temp_c;
    temp_rate_     = 0.0f;

    memset(window_, 0, sizeof(window_));

    // Initial state: [alt = 0m, vel = 0m/s, accel_bias = 0m/s^2]
    x_[0] = 0.0f;
    x_[1] = 0.0f;
    x_[2] = 0.0f;

    // Initial covariance matrix
    memset(P_, 0, sizeof(P_));
    P_[0][0] = 0.25f;
    P_[1][1] = 0.10f;
    P_[2][2] = 0.02f;

    initialized_ = true;
}

void CnnEkfFilter::resetReference(float current_pressure_hpa) {
    ref_altitude_  = pressureToAltitude(current_pressure_hpa);
    prev_baro_alt_ = 0.0f;
    x_[0] = 0.0f;
    x_[1] = 0.0f;
}

float CnnEkfFilter::runCnnInference() const {
    constexpr int L1_LEN = CNN_WINDOW_SIZE - 2; // 14
    constexpr int L2_LEN = L1_LEN - 2;          // 12

    float conv1_out[L1_LEN][4];
    float conv2_out[L2_LEN][4];

    // 1. Conv1D Layer 1 + ReLU
    for (int t = 0; t < L1_LEN; ++t) {
        for (int oc = 0; oc < 4; ++oc) {
            float sum = CONV1_B[oc];
            for (int k = 0; k < 3; ++k) {
                for (int ic = 0; ic < 3; ++ic) {
                    sum += window_[t + k][ic] * CONV1_W[oc][k][ic];
                }
            }
            conv1_out[t][oc] = relu(sum);
        }
    }

    // 2. Conv1D Layer 2 + ReLU
    for (int t = 0; t < L2_LEN; ++t) {
        for (int oc = 0; oc < 4; ++oc) {
            float sum = CONV2_B[oc];
            for (int k = 0; k < 3; ++k) {
                for (int ic = 0; ic < 4; ++ic) {
                    sum += conv1_out[t + k][ic] * CONV2_W[oc][k][ic];
                }
            }
            conv2_out[t][oc] = relu(sum);
        }
    }

    // 3. Global Average Pooling + Dense Output
    float raw_score = DENSE_B;
    for (int oc = 0; oc < 4; ++oc) {
        float channel_sum = 0.0f;
        for (int t = 0; t < L2_LEN; ++t) {
            channel_sum += conv2_out[t][oc];
        }
        raw_score += (channel_sum / static_cast<float>(L2_LEN)) * DENSE_W[oc];
    }

    // Map score smoothly to covariance multiplier alpha in [1.0, 25.0]
    float scaled = raw_score * 0.45f - 0.30f;
    float norm_activation = (scaled > 0.0f) ? tanhf(scaled) : 0.0f;
    return 1.0f + 24.0f * norm_activation;
}

FilterOutput CnnEkfFilter::update(float pressure_hpa, float temp_c, float accel_z_ms2, float dt) {
    if (!initialized_) {
        begin(pressure_hpa, temp_c);
    }

    // 1. Compute relative barometric altitude and smoothed thermal gradient
    float raw_baro_alt = pressureToAltitude(pressure_hpa) - ref_altitude_;
    float baro_delta   = raw_baro_alt - prev_baro_alt_;
    prev_baro_alt_     = raw_baro_alt;

    float prev_temp = filtered_temp_;
    filtered_temp_  = 0.85f * filtered_temp_ + 0.15f * temp_c;
    float inst_rate = (filtered_temp_ - prev_temp) / dt;
    temp_rate_      = 0.80f * temp_rate_ + 0.20f * inst_rate;

    // 2. EKF Prediction Step (kinematic propagation using MPU-6050 vertical accel)
    float u = accel_z_ms2 - x_[2]; // Bias-corrected vertical acceleration
    float z_pred = x_[0] + x_[1] * dt + 0.5f * u * dt * dt;
    float v_pred = x_[1] + u * dt;
    float b_pred = x_[2];

    // State transition Jacobian F = [[1, dt, -0.5*dt^2], [0, 1, -dt], [0, 0, 1]]
    float dt2 = 0.5f * dt * dt;
    float F[3][3] = {
        {1.0f, dt,   -dt2},
        {0.0f, 1.0f, -dt},
        {0.0f, 0.0f, 1.0f}
    };

    // P_pred = F * P * F^T + Q
    float FP[3][3] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            FP[i][j] = F[i][0] * P_[0][j] + F[i][1] * P_[1][j] + F[i][2] * P_[2][j];
        }
    }

    float P_pred[3][3] = {0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P_pred[i][j] = FP[i][0] * F[j][0] + FP[i][1] * F[j][1] + FP[i][2] * F[j][2];
        }
    }
    P_pred[0][0] += EKF_Q_ALT * dt;
    P_pred[1][1] += EKF_Q_VEL * dt;
    P_pred[2][2] += 0.0005f * dt; // Slow accelerometer bias random walk

    // 3. Push normalized features into sliding window & run 1D-CNN
    float innovation = raw_baro_alt - z_pred;
    memmove(&window_[0][0], &window_[1][0], (CNN_WINDOW_SIZE - 1) * 3 * sizeof(float));
    window_[CNN_WINDOW_SIZE - 1][0] = baro_delta * 10.0f;
    window_[CNN_WINDOW_SIZE - 1][1] = innovation;
    window_[CNN_WINDOW_SIZE - 1][2] = temp_rate_;

    float alpha = runCnnInference();
    float R_k   = alpha * EKF_R_BASE;

    // 4. EKF Measurement Update (H = [1, 0, 0])
    float S    = P_pred[0][0] + R_k;
    float K[3] = {
        P_pred[0][0] / S,
        P_pred[1][0] / S,
        P_pred[2][0] / S
    };

    x_[0] = z_pred + K[0] * innovation;
    x_[1] = v_pred + K[1] * innovation;
    x_[2] = b_pred + K[2] * innovation;

    // Covariance update: P = (I - K * H) * P_pred
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            P_[i][j] = P_pred[i][j] - K[i] * P_pred[0][j];
        }
    }

    // 5. Pack output
    FilterOutput out;
    out.altitude      = x_[0];
    out.velocity_z    = x_[1];
    out.raw_baro_alt  = raw_baro_alt;
    out.temp_c        = filtered_temp_;
    out.r_scale       = static_cast<uint8_t>(roundf(alpha * 10.0f));
    out.thermal_alert = (fabsf(temp_rate_) > THERMAL_SPIKE_THRESH);
    return out;
}
