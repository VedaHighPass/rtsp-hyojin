#pragma once
#include <cstddef>
#include <cstdint>
#include <arpa/inet.h>
#include <rtp_header.hpp>
class RtpPacket
{
private:
    RtpHeader header;
    uint8_t RTP_Payload[FU_SIZE + MAX_RTP_DATA_SIZE]{0};
public:
    explicit RtpPacket(const RtpHeader &rtpHeader);
    RtpPacket(const RtpPacket &) = default;
    ~RtpPacket() = default;
    void load_data(const uint8_t *data, int64_t dataSize, int64_t bias = 0);
    int64_t rtp_sendto(int sockfd, int64_t _bufferLen, int flags, const sockaddr *to);
    RtpHeader& get_header() { return this->header; }
    uint8_t *get_payload() { return this->RTP_Payload; }
};
#pragma pack()
