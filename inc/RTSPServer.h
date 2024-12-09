#ifndef RTSP_SERVER_H
#define RTSP_SERVER_H

#include <thread>
#include "TCPHandler.h"

class RTSPServer {
public:
    RTSPServer();
    ~RTSPServer();
    void start();

private:
    void clientHandler();
    std::thread serverThread;
    bool stopFlag = false;
};

#endif // RTSP_SERVER_H

