#ifndef RTSP_TCPHANDLER_H
#define RTSP_TCPHANDLER_H

#include <string>
#include <arpa/inet.h>
#include <unordered_map>

class TCPHandler {
public:
    TCPHandler(const TCPHandler&) = delete;
    TCPHandler& operator=(const TCPHandler&) = delete;

    static TCPHandler& GetInstance() {
        static TCPHandler instance;
        return instance;
    };

    void CreateTCPSocket();

    std::pair<int, std::string> AcceptClientConnection();

    std::string ReceiveRTSPRequest(int sessionId);

    void SendRTSPResponse(int clientSocket, std::string& response);

    int& GetTCPSocket();

    sockaddr_in& GetTCPAddr();


private:
    TCPHandler();
    ~TCPHandler();

    int tcpPort;
    int tcpSocket;

    sockaddr_in tcpAddr;

    std::unordered_map<int, int> socketTable;
};

#endif //RTSP_TCPHANDLER_H