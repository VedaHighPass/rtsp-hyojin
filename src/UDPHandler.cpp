#include "UDPHandler.h"
#include "ClientSession.h"
#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

UDPHandler::UDPHandler(const std::shared_ptr<Info>& client)
    : client(client), rtpSocket(-1), rtcpSocket(-1) {}

UDPHandler::~UDPHandler() {
    if (rtpSocket != -1) close(rtpSocket);
    if (rtcpSocket != -1) close(rtcpSocket);
}

bool UDPHandler::CreateUDPSocket() {
    rtpSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (rtpSocket == -1) {
        std::cerr << "rtp 소켓 생성 실패" << std::endl;
        return false;
    }

    rtcpSocket = socket(AF_INET, SOCK_DGRAM, 0);
    if (rtcpSocket == -1) {
        std::cerr << "rtcp 소켓 생성 실패" << std::endl;
        return false;
    }

    memset(&rtpAddr, 0, sizeof(rtpAddr));
    rtpAddr.sin_family = AF_INET;
    rtpAddr.sin_port = htons(client->rtpPort);
    inet_pton(AF_INET, client->ip.c_str(), &rtpAddr.sin_addr);

    memset(&rtcpAddr, 0, sizeof(rtcpAddr));
    rtcpAddr.sin_family = AF_INET;
    rtcpAddr.sin_port = htons(client->rtcpPort);
    inet_pton(AF_INET, client->ip.c_str(), &rtcpAddr.sin_addr);

    return true;
}

// void UDPHandler::SendSenderReport(Protos::SenderReport* senderReport, size_t srSize) {
//     int sentBytes = sendto(rtcpSocket, senderReport, srSize, 0, (struct sockaddr*)&rtcpAddr, sizeof(rtcpAddr));
//     if (sentBytes == -1) {
//         std::cerr << "Error: fail to send RTCP packet" << std::endl;
//         exit(1);
//     }
// }

int& UDPHandler::GetRTPSocket() { return rtpSocket; }
int& UDPHandler::GetRTCPSocket() { return rtcpSocket; }

sockaddr_in& UDPHandler::GetRTPAddr() { return rtpAddr; }
sockaddr_in& UDPHandler::GetRTCPAddr() { return rtcpAddr; }
