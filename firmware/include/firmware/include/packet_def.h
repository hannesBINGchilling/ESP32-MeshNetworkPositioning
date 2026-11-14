#pragma once

#include <stdint.h>

// Alert flag definitions
#define ALERT_NONE            0
#define ALERT_PANIC_BTN       1
#define ALERT_THERMAL_SPIKE   2

// Force 1-byte alignment so packet size is strictly 24 bytes across architectures
#pragma pack(push, 1)
struct MeshPacket {
    uint32_t node_id;      // Unique ESP32 chip ID of the origin node
    uint32_t seq_num;      // Monotonically increasing packet counter
    float    altitude;     // Filtered relative altitude (m)
    float    velocity_z;   // Estimated vertical velocity (m/s)
    float    temp_c;       // Ambient temperature (°C)
    uint8_t  hop_count;    // Number of relays traversed (0 = direct)
    uint8_t  ttl;          // Remaining Time-To-Live hops
    uint8_t  alert_flag;   // 0 = Normal, 1 = Panic button, 2 = Thermal spike
    uint8_t  r_scale;      // 1D-CNN noise multiplier (10 = 1.0x baseline R)
};
#pragma pack(pop)

// Compile-time guard to verify exact 24-byte wire format
static_assert(sizeof(MeshPacket) == 24, "MeshPacket size mismatch: expected 24 bytes");
