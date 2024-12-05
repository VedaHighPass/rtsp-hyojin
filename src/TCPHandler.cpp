#include "TCPHandler.h"
#include "global.h"

#include <string>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

TCPHandler::TCPHandler(): tcpPort(g_serverRtpPort), tcpSocket(-1) {
    CreateTCPSocket();
}

TCPHandler::~TCPHandler() {
    if (tcpSocket != -1) close(tcpSocket);
}

void TCPHandler::CreateTCPSocket() {
    tcpSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (tcpSocket == -1) {
        std::cerr << "Error: fail to create TCP socket" << std::endl;
        exit(1);
    }

    memset(&tcpAddr, 0, sizeof(tcpAddr));
    tcpAddr.sin_family = AF_INET;
    tcpAddr.sin_addr.s_addr = INADDR_ANY;
    tcpAddr.sin_port = htons(tcpPort);

    if (bind(tcpSocket, (struct sockaddr*)&tcpAddr, sizeof(tcpAddr)) == -1) {
        std::cerr << "Error: fail to bind TCP socket" << std::endl;
        exit(1);
    }

    if (listen(tcpSocket, 10) == -1) {
        std::cerr << "Error: fail to listen TCP socket" << std::endl;
        exit(1);
    }
}

std::pair<int, std::string> TCPHandler::AcceptClientConnection() {
    socklen_t clientAddrLen = sizeof(tcpAddr);
    int clientSocket = accept(tcpSocket, (sockaddr*)&tcpAddr, &clientAddrLen);

    if (clientSocket == -1) {
        std::cerr << "Error: fail to connect client" << std::endl;
        exit(1);
    }

    char clientIP[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &tcpAddr.sin_addr, clientIP, INET_ADDRSTRLEN);
    
    return { clientSocket, clientIP };
}

std::string TCPHandler::ReceiveRTSPRequest(int clientSocket) {
    char buffer[1024];
    memset(buffer, 0, sizeof(buffer));

    int receivedBytes = recv(clientSocket, buffer, sizeof(buffer) - 1, 0);

    if (receivedBytes <= 0) {
        std::cerr << "Error: fail to recv RTSP request" << std::endl;
        return nullptr;
    }

    std::cout << "rtsp 요청:\n" << buffer << std::endl;
    return {buffer};
}

void TCPHandler::SendRTSPResponse(int clientSocket, std::string& response) {
    int sentBytes = send(clientSocket, response.c_str(), response.size(), 0);
    if (sentBytes == -1) {
        std::cerr << "Error: fail to send RTSP response" << std::endl;
    }
}

int& TCPHandler::GetTCPSocket() { return tcpSocket; }

sockaddr_in& TCPHandler::GetTCPAddr() { return tcpAddr; }
