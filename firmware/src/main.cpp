#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "config.h"
#include "packet_def.h"
#include "cnn_ekf_filter.h"
#include "mesh_network.h"

Adafruit_BMP280 bmp;
Adafruit_MPU6050 mpu;
CnnEkfFilter nav_filter;
MeshNetwork mesh_net;

float gravity_z_bias = 9.80665f;
uint32_t last_sample_ms = 0;
uint32_t last_broadcast_ms = 0;
FilterOutput latest_out;

void calibrateStationaryGravity() {
    Serial.println("[INIT] Keep device still for 1s to calibrate vertical gravity...");
    float sum_az = 0.0f;
    const int samples = 50;
    for (int i = 0; i < samples; ++i) {
        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        sum_az += a.acceleration.z;
        delay(20);
    }
    gravity_z_bias = sum_az / static_cast<float>(samples);
    Serial.printf("[INIT] Gravity baseline calibrated: %.3f m/s^2\n", gravity_z_bias);
}

void setup() {
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== ESP32 MeshNav (1D-CNN + EKF) Starting ===");

    pinMode(PANIC_BTN_PIN, INPUT_PULLUP);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(400000);

    // 1. Initialize BMP280 Barometer
    if (!bmp.begin(BMP280_ADDR)) {
        Serial.println("[ERROR] BMP280 not found! Check wiring or try BMP280_ADDR 0x77 in config.h.");
        while (true) {
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
            delay(200);
        }
    }
    bmp.setSampling(Adafruit_BMP280::MODE_NORMAL,
                    Adafruit_BMP280::SAMPLING_X2,
                    Adafruit_BMP280::SAMPLING_X16,
                    Adafruit_BMP280::FILTER_X4,
                    Adafruit_BMP280::STANDBY_MS_1);

    // 2. Initialize MPU-6050 IMU
    if (!mpu.begin(MPU6050_ADDR)) {
        Serial.println("[ERROR] MPU-6050 not found! Check SDA (GPIO 21) and SCL (GPIO 22).");
        while (true) {
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
            delay(200);
        }
    }
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

    // 3. Calibrate baseline & initialize 1D-CNN + EKF filter
    calibrateStationaryGravity();
    float init_p = bmp.readPressure() / 100.0f; // Pa -> hPa
    float init_t = bmp.readTemperature();
    nav_filter.begin(init_p, init_t);

    // 4. Start Ad-Hoc Mesh Network
    mesh_net.begin();
    Serial.println("[READY] Running 50Hz sensor fusion & 5Hz mesh broadcast. Send 'z' to zero altitude.");

    last_sample_ms = millis();
    last_broadcast_ms = millis();
}

void loop() {
    mesh_net.update();
    uint32_t now = millis();

    // Allow manual floor re-zeroing via Serial Monitor ('z')
    if (Serial.available() > 0) {
        char cmd = static_cast<char>(Serial.read());
        if (cmd == 'z' || cmd == 'Z') {
            float p_hpa = bmp.readPressure() / 100.0f;
            nav_filter.resetReference(p_hpa);
            Serial.println("[CAL] Relative altitude zeroed to current floor.");
        }
    }

    // 50 Hz Sensor Sampling & 1D-CNN + EKF State Fusion
    if (now - last_sample_ms >= (1000 / SENSOR_SAMPLE_HZ)) {
        float dt = (now - last_sample_ms) / 1000.0f;
        last_sample_ms = now;

        float pressure_hpa = bmp.readPressure() / 100.0f;
        float temp_c = bmp.readTemperature();

        sensors_event_t a, g, temp;
        mpu.getEvent(&a, &g, &temp);
        float linear_az = a.acceleration.z - gravity_z_bias;

        latest_out = nav_filter.update(pressure_hpa, temp_c, linear_az, dt);
    }

    // 5 Hz Mesh Broadcast & Local Serial Telemetry
    if (now - last_broadcast_ms >= BROADCAST_INTERVAL_MS) {
        last_broadcast_ms = now;

        uint8_t alert = ALERT_NONE;
        if (digitalRead(PANIC_BTN_PIN) == LOW) {
            alert = ALERT_PANIC_BTN;
        } else if (latest_out.thermal_alert) {
            alert = ALERT_THERMAL_SPIKE;
        }

        mesh_net.broadcastLocalState(latest_out.altitude,
                                     latest_out.velocity_z,
                                     latest_out.temp_c,
                                     alert,
                                     latest_out.r_scale);

        // Heartbeat blink on onboard LED
        digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));

        Serial.printf("[LOCAL] Alt:%.2fm (Raw:%.2fm) | Vel:%.2fm/s | Temp:%.1fC | R_scale:%.1fx | Nodes:%u\n",
                      latest_out.altitude,
                      latest_out.raw_baro_alt,
                      latest_out.velocity_z,
                      latest_out.temp_c,
                      latest_out.r_scale / 10.0f,
                      mesh_net.getNeighborCount());
    }
}
