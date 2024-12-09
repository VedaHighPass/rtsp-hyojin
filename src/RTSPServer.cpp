#include "RTSPServer.h"
#include <iostream>
#include "ClientSession.h"
#include "TCPHandler.h"
#
RTSPServer::RTSPServer() : stopFlag(false) {}

RTSPServer::~RTSPServer() {
    stop();
}

void RTSPServer::start() {
    stopFlag = false;
    serverThread = std::thread(&RTSPServer::clientHandler, this);
}

void RTSPServer::stop() {
    stopFlag = true;
    if (serverThread.joinable()) {
        serverThread.join();
    }
}

void RTSPServer::clientHandler() {
    while (!stopFlag) {
        std::pair<int, std::string> newClient = TCPHandler::GetInstance().AcceptClientConnection();

        std::cout << "Client connected" << std::endl;

        ClientSession* clientSession = new ClientSession(newClient);
        clientSession->StartRequestHandlerThread();
    }
}

