#include <rtp_packet.hpp>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <arpa/inet.h>
#include <netinet/in.h>
RtpPacket::RtpPacket(const RtpHeader &rtpHeader) : header(rtpHeader) {}
void RtpPacket::load_data(const uint8_t *data, const int64_t dataSize, const int64_t bias)
{
    memcpy(this->RTP_Payload + bias, data, std::min(dataSize, static_cast<int64_t>(sizeof(this->RTP_Payload) - bias)));
}
int64_t RtpPacket::rtp_sendto(int sockfd, const int64_t _bufferLen, const int flags, const sockaddr *to)
{
    auto sentBytes = sendto(sockfd, this, _bufferLen, flags, to, sizeof(sockaddr));
    header.set_seq(header.get_seq() + 1);
    return sentBytes;
}
