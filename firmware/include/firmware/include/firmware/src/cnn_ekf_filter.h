#pragma once

#include <stdint.h>
#include "config.h"

struct FilterOutput {
    float   altitude;       // Denoised relative altitude (m)
    float   velocity_z;     // Vertical velocity estimate (m/s)
    float   raw_baro_alt;   // Unfiltered relative barometric altitude (m)
    float   temp_c;         // Smoothed temperature (°C)
    uint8_t r_scale;        // Dynamic noise multiplier (10 = 1.0x baseline R)
    bool    thermal_alert;  // True if rapid temperature spike is detected
};

class CnnEkfFilter {
public:
    CnnEkfFilter();

    // Call once at startup to set the 0.0m ground reference altitude
    void begin(float initial_pressure_hpa, float initial_temp_c);

    // Run one fusion step at 50 Hz (accel_z_ms2 should have gravity removed)
    FilterOutput update(float pressure_hpa, float temp_c, float accel_z_ms2, float dt = SENSOR_DT);

    // Re-zero relative altitude to current floor/ground level
    void resetReference(float current_pressure_hpa);

private:
    float pressureToAltitude(float pressure_hpa) const;
    float runCnnInference() const;

    bool  initialized_;
    float ref_altitude_;     // Ground reference altitude (m)
    float prev_baro_alt_;    // Previous barometric reading (m)
    float filtered_temp_;    // EMA-smoothed temperature (°C)
    float temp_rate_;        // Temperature change rate (°C/s)

    // Sliding window buffer: [time_step][3 feature channels]
    // Ch 0: Baro step delta | Ch 1: Baro-EKF innovation | Ch 2: Temp gradient
    float window_[CNN_WINDOW_SIZE][3];

    // 3-state EKF state vector: [altitude (m), velocity_z (m/s), accel_bias (m/s^2)]
    float x_[3];
    float P_[3][3];
};
