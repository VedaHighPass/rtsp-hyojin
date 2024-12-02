#ifndef RTSP_CLIENTSESSION_H
#define RTSP_CLIENTSESSION_H

#include <map>
#include <queue>
#include <mutex>
#include <chrono>
#include <string>
#include <iostream>
#include <memory>

struct Info {
    int id;
    int version;
    int tcpSocket;
    int rtpPort;
    int rtcpPort;
    std::string ip;
    std::string state;
};

class RequestHandler;
class UDPHandler;
class MediaStreamHandler;

class ClientSession {
public:
    ClientSession(const std::pair<int, std::string>& newClient);
    
    void StartRequestHandlerThread();

    int GetSessionId() const;

    int GetVersion() const;

    std::string GetState() const;

    std::pair<int, int> GetPort() const;

    void SetState(const std::string& newState);

private:
    
    std::shared_ptr<Info> info;

    // std::mutex mtx;
    
    RequestHandler* requestHandler;
};

#endif //RTSP_CLIENTSESSION_H