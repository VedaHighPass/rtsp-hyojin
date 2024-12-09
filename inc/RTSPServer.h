#ifndef RTSP_SERVER_H
#define RTSP_SERVER_H

#include <thread>
#include "TCPHandler.h"

class RTSPServer {
public:
    RTSPServer();
    ~RTSPServer();

    void start();
    void stop();

private:
    void clientHandler();
    std::thread serverThread;
    bool stopFlag;
};

#endif // RTSP_SERVER_H

