#include "mesh_network.h"
#include <painlessMesh.h>
#include <Arduino.h>
#include <string.h>

namespace {
painlessMesh g_mesh;
MeshNetwork* g_instance = nullptr;

// Encode 24-byte struct as 48-char hex string to prevent null-byte truncation in String
String packetToHex(const MeshPacket& pkt) {
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(&pkt);
    char hex_buf[sizeof(MeshPacket) * 2 + 1];
    for (size_t i = 0; i < sizeof(MeshPacket); ++i) {
        snprintf(&hex_buf[i * 2], 3, "%02X", raw[i]);
    }
    hex_buf[sizeof(MeshPacket) * 2] = '\0';
    return String(hex_buf);
}

bool hexToPacket(const String& hex, MeshPacket& out_pkt) {
    if (hex.length() != sizeof(MeshPacket) * 2) {
        return false;
    }
    uint8_t* raw = reinterpret_cast<uint8_t*>(&out_pkt);
    for (size_t i = 0; i < sizeof(MeshPacket); ++i) {
        char byte_str[3] = { hex[i * 2], hex[i * 2 + 1], '\0' };
        raw[i] = static_cast<uint8_t>(strtoul(byte_str, nullptr, 16));
    }
    return true;
}
} // namespace

MeshNetwork::MeshNetwork() : local_seq_(0), seen_idx_(0) {
    memset(seen_cache_, 0, sizeof(seen_cache_));
    memset(relay_queue_, 0, sizeof(relay_queue_));
    g_instance = this;
}

void MeshNetwork::begin() {
    g_mesh.setDebugMsgTypes(ERROR | STARTUP);
    g_mesh.init(MESH_PREFIX, MESH_PASSWORD, MESH_PORT, WIFI_AP_STA, MESH_CHANNEL);

    g_mesh.onReceive([](uint32_t from, String& msg) {
        (void)from;
        if (!g_instance) return;

        MeshPacket pkt;
        if (!hexToPacket(msg, pkt)) return;

        // 1. Drop duplicates already seen in sliding cache
        if (g_instance->isDuplicate(pkt.node_id, pkt.seq_num)) {
            return;
        }
        g_instance->markSeen(pkt.node_id, pkt.seq_num);

        // Print received remote node telemetry to Serial Monitor
        Serial.printf("[MESH RX] Node:%08X | Seq:%lu | Alt:%.2fm | Vel:%.2fm/s | Temp:%.1fC | Hops:%u | Alert:%u\n",
                      pkt.node_id,
                      static_cast<unsigned long>(pkt.seq_num),
                      pkt.altitude,
                      pkt.velocity_z,
                      pkt.temp_c,
                      pkt.hop_count,
                      pkt.alert_flag);

        // 2. Forward packet if TTL allows
        if (pkt.ttl > 1) {
            pkt.ttl -= 1;
            pkt.hop_count += 1;
            g_instance->enqueueRelay(pkt);
        }
    });

    g_mesh.onChangedConnections([]() {
        Serial.printf("[MESH] Topology updated. Connected neighbors: %u\n",
                      static_cast<unsigned int>(g_mesh.getNodeList().size()));
    });
}

void MeshNetwork::update() {
    g_mesh.update();

    // Process non-blocking randomized backoff relay queue
    uint32_t now = millis();
    for (uint8_t i = 0; i < RELAY_QUEUE_SIZE; ++i) {
        if (relay_queue_[i].active && static_cast<int32_t>(now - relay_queue_[i].send_at_ms) >= 0) {
            sendPacket(relay_queue_[i].pkt);
            relay_queue_[i].active = false;
        }
    }
}

void MeshNetwork::broadcastLocalState(float altitude, float velocity_z, float temp_c,
                                      uint8_t alert_flag, uint8_t r_scale) {
    MeshPacket pkt;
    pkt.node_id    = g_mesh.getNodeId();
    pkt.seq_num    = ++local_seq_;
    pkt.altitude   = altitude;
    pkt.velocity_z = velocity_z;
    pkt.temp_c     = temp_c;
    pkt.hop_count  = 0;
    pkt.ttl        = DEFAULT_TTL;
    pkt.alert_flag = alert_flag;
    pkt.r_scale    = r_scale;

    markSeen(pkt.node_id, pkt.seq_num);
    sendPacket(pkt);
}

uint16_t MeshNetwork::getNeighborCount() const {
    return static_cast<uint16_t>(g_mesh.getNodeList().size());
}

bool MeshNetwork::isDuplicate(uint32_t node_id, uint32_t seq_num) const {
    for (uint8_t i = 0; i < SEEN_CACHE_SIZE; ++i) {
        if (seen_cache_[i].node_id == node_id && seen_cache_[i].seq_num == seq_num) {
            return true;
        }
    }
    return false;
}

void MeshNetwork::markSeen(uint32_t node_id, uint32_t seq_num) {
    seen_cache_[seen_idx_].node_id = node_id;
    seen_cache_[seen_idx_].seq_num = seq_num;
    seen_idx_ = (seen_idx_ + 1) % SEEN_CACHE_SIZE;
}

void MeshNetwork::enqueueRelay(const MeshPacket& pkt) {
    for (uint8_t i = 0; i < RELAY_QUEUE_SIZE; ++i) {
        if (!relay_queue_[i].active) {
            relay_queue_[i].active     = true;
            relay_queue_[i].send_at_ms = millis() + random(10, 31); // 10-30 ms CSMA jitter
            relay_queue_[i].pkt        = pkt;
            return;
        }
    }
}

void MeshNetwork::sendPacket(const MeshPacket& pkt) {
    String payload = packetToHex(pkt);
    g_mesh.sendBroadcast(payload);
}
