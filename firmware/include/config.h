#pragma once


// Hardware Pinout (ESP32 DevKit V1)

#define I2C_SDA_PIN           21
#define I2C_SCL_PIN           22
#define PANIC_BTN_PIN         4     // Active-LOW push button to GND
#define STATUS_LED_PIN        2     // Onboard blue LED


//  I2C Sensor Addresses

// Note: Some BMP280 breakout boards use 0x77 if SDO is pulled HIGH.
#define BMP280_ADDR           0x76
#define MPU6050_ADDR          0x68


// Ad-Hoc Mesh Network Settings

#define MESH_PREFIX           "MeshNav_Node"
#define MESH_PASSWORD         "meshnav1234"
#define MESH_PORT             5555
#define MESH_CHANNEL          6

#define DEFAULT_TTL           5     // Max relay hops per packet
#define BROADCAST_INTERVAL_MS 200   // Broadcast state every 200 ms (5 Hz)
#define SEEN_CACHE_SIZE       32    // Number of recent packet IDs kept for deduplication


// Sensor Sampling & 1D-CNN + EKF Tuning

#define SENSOR_SAMPLE_HZ      50
#define SENSOR_DT             0.02f // 20 ms sampling period

#define CNN_WINDOW_SIZE       16    // 16 samples = 0.32s sliding window
#define EKF_Q_ALT             0.01f // Process noise variance (altitude)
#define EKF_Q_VEL             0.08f // Process noise variance (vertical velocity)
#define EKF_R_BASE            0.35f // Baseline barometer measurement noise variance (m^2)

#define THERMAL_SPIKE_THRESH  1.2f  // Temp rise rate (°C/s) to trigger thermal alert
#define SEA_LEVEL_HPA         1013.25f
