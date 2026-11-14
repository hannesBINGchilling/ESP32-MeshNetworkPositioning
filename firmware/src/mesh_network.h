#pragma once

#include <stdint.h>
#include "config.h"
#include "packet_def.h"

class MeshNetwork {
public:
    MeshNetwork();

    // Initialize Wi-Fi mesh radio and packet callbacks
    void begin();

    // Must be called frequently in loop() to handle mesh routing and delayed relays
    void update();

    // Broadcast this node's latest filtered state to all reachable neighbors
    void broadcastLocalState(float altitude, float velocity_z, float temp_c,
                             uint8_t alert_flag, uint8_t r_scale);

    // Return number of currently connected mesh nodes (excluding self)
    uint16_t getNeighborCount() const;

private:
    struct SeenPacket {
        uint32_t node_id;
        uint32_t seq_num;
    };

    struct RelayItem {
        bool       active;
        uint32_t   send_at_ms;
        MeshPacket pkt;
    };

    static constexpr uint8_t RELAY_QUEUE_SIZE = 8;

    uint32_t   local_seq_;
    uint8_t    seen_idx_;
    SeenPacket seen_cache_[SEEN_CACHE_SIZE];
    RelayItem  relay_queue_[RELAY_QUEUE_SIZE];

    bool isDuplicate(uint32_t node_id, uint32_t seq_num) const;
    void markSeen(uint32_t node_id, uint32_t seq_num);
    void enqueueRelay(const MeshPacket& pkt);
    void sendPacket(const MeshPacket& pkt);
};
