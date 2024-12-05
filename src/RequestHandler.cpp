#include "utils.h"
#include "RequestHandler.h"
#include "TCPHandler.h"
#include "ClientSession.h"
#include "MediaStreamHandler.h"
#include "UDPHandler.h"
#include "global.h"

#include <iostream>
#include <string>
#include <sstream>
#include <thread>

RequestHandler::RequestHandler(const std::shared_ptr<Info>& client)
        : client(client) {}

void RequestHandler::HandleRequest() {
     std::cout << "create Client Session" << std::endl;

    while (true) {
        std::cout << "recv wait id :" << client->tcpSocket << std::endl;
        std::string request = TCPHandler::GetInstance().ReceiveRTSPRequest(client->tcpSocket);
        if (request.empty()) {
            std::cerr << "Invalid RTSP request received." << std::endl;
            return;
        }

        std::string method = ParseMethod(request);

        int cseq = ParseCSeq(request);
        if (cseq == -1) {
            std::cerr << "CSeq parsing failed." << std::endl;
            return;
        }

        std::cout << "METHOD = "<<method<<std::endl;
        if (method == "OPTIONS") {
            HandleOptionsRequest(cseq);
        } else if (method == "DESCRIBE") {
            HandleDescribeRequest(request, cseq);
        } else if (method == "SETUP") {
            HandleSetupRequest(request, cseq);
        } else if (method == "PLAY") {
            HandlePlayRequest(cseq);
        } else if (method == "PAUSE") {
            HandlePauseRequest(cseq);
        } else if (method == "TEARDOWN") {
            HandleTeardownRequest(cseq);
        } else {
            std::cerr << "Unsupported RTSP method: " << method << std::endl;
        }
    }
}

std::string RequestHandler::ParseMethod(const std::string& request) {
    std::istringstream requestStream(request);
    std::string method;
    requestStream >> method;
    return method;
}

int RequestHandler::ParseCSeq(const std::string& request) {
    std::istringstream requestStream(request);
    std::string line;
    while (getline(requestStream, line)) {
        if (line.find("CSeq") != std::string::npos) {
            std::istringstream lineStream(line);
            std::string label;
            int cseq;
            lineStream >> label >> cseq;
            return cseq;
        }
    }
    return -1; // CSeq not found
}

std::pair<int, int> RequestHandler::ParsePorts(const std::string& request) {
    std::istringstream requestStream(request);
    std::string line;
    while (getline(requestStream, line)) {
        if (line.find("client_port=") != std::string::npos) {
            std::istringstream lineStream(line);
            std::string label;

            while (getline(lineStream, label, '/')) {
                std::string portRange;
                getline(lineStream, portRange);
                size_t eqPos = portRange.find('=') + 1;
                size_t dashPos = portRange.find('-');

                if (dashPos != std::string::npos) {

                    int rtpPort = stoi(portRange.substr(eqPos, dashPos - eqPos));
                    int rtcpPort = stoi(portRange.substr(dashPos + 1));
		            return {rtpPort, rtcpPort};
                }
            }
        }
    }
    return {-1, -1};
}

bool RequestHandler::ParseAccept(const std::string& request) {
    std::istringstream requestStream(request);
    std::string line;
    while (getline(requestStream, line))
        if (line.find("application/sdp") != std::string::npos)
            return true;
    return false;
}

void RequestHandler::HandleOptionsRequest(int cseq) {
    std::string response = "RTSP/1.0 200 OK\r\n"
                           "CSeq: " + std::to_string(cseq) + "\r\n"
                           "Public: DESCRIBE, SETUP, TEARDOWN, PLAY, PAUSE\r\n"
                           "\r\n";
    TCPHandler::GetInstance().SendRTSPResponse(client->tcpSocket, response);
}

void RequestHandler::HandleDescribeRequest(const std::string& request, int cseq) {
    std::string ip = utils::GetIP();
    std::string sdp = "";
    std::string response = "";

    if (ParseAccept(request)) {
        response = "RTSP/1.0 200 OK\r\n";
        sdp = "v=0\r\n"
          "o=- " + std::to_string(client->id) + " " + std::to_string(client->version) +
          " IN IP4 " + ip + "\r\n"
          "s=H264 Video Stream\r\n"
          "c=IN IP4 " + ip + "\r\n"
          "t=0 0\r\n"
          "m=video " + std::to_string(client->rtpPort) + " RTP/AVP 96\r\n"
          "a=rtpmap:96 H264/90000\r\n"
          "a=control:track0\r\n";
    } else {
        response = "RTSP/1.0 406 Not Acceptable\r\n";
    }

    response += "CSeq: " + std::to_string(cseq) + "\r\n"
                "Content-Base: rtsp://" + ip + ":" + std::to_string(g_serverRtpPort) +"/\r\n"
                "Content-Type: application/sdp\r\n"
                "Content-Length: " + std::to_string(sdp.size()) + "\r\n"
                "\r\n" + sdp;

    TCPHandler::GetInstance().SendRTSPResponse(client->tcpSocket, response);
}


void RequestHandler::HandleSetupRequest(const std::string& request, int cseq) {
    client->state = "SETUP";

    auto ports = ParsePorts(request);
    if (ports.first < 0 || ports.second < 0) {
        std::cerr << "not found IP or Port in SETUP" << std::endl;
        return;
    }

    client->rtpPort = ports.first;
    client->rtcpPort = ports.second;
    std::string response = "RTSP/1.0 200 OK\r\n"
                           "CSeq: " + std::to_string(cseq) + "\r\n"
                           "Transport: RTP/AVP;unicast;client_port="
                           + std::to_string(client->rtpPort) + "-"
                           + std::to_string(client->rtcpPort) + "\r\n"
                           "Session: " + std::to_string(client->id)
                           + "\r\n"
                             "\r\n";
    TCPHandler::GetInstance().SendRTSPResponse(client->tcpSocket, response);

    mediaStreamHandler = new MediaStreamHandler();
    mediaStreamHandler->udpHandler = new UDPHandler(client);
    mediaStreamHandler->udpHandler->CreateUDPSocket();
    std::thread mediaStreamThread(&MediaStreamHandler::HandleMediaStream, mediaStreamHandler);
    //TODO : 스레드 우선순위 높이기
    mediaStreamThread.detach();
}

void RequestHandler::HandlePlayRequest(int cseq) {
    std::cout<< "PLAY REQUEST START\n";
    client->state = "PLAY";

    std::string response = "RTSP/1.0 200 OK\r\n"
                           "CSeq: " + std::to_string(cseq) + "\r\n"
                           "Session: " + std::to_string(client->id)
                           + "\r\n"
                             "\r\n";
    TCPHandler::GetInstance().SendRTSPResponse(client->tcpSocket, response);

    mediaStreamHandler->SetCmd("PLAY");
}

void RequestHandler::HandlePauseRequest(int cseq) {
    client->state = "PAUSE";

    std::string response = "RTSP/1.0 200 OK\r\n"
                           "CSeq: " + std::to_string(cseq) + "\r\n"
                           "Session: " + std::to_string(client->id)
                           + "\r\n"
                             "\r\n";

    TCPHandler::GetInstance().SendRTSPResponse(client->tcpSocket, response);

    mediaStreamHandler->SetCmd("PAUSE");
}

void RequestHandler::HandleTeardownRequest(int cseq) {
    client->state = "TEARDOWN";

    std::string response = "RTSP/1.0 200 OK\r\n"
                           "CSeq: " + std::to_string(cseq) + "\r\n"
                           "Session: " + std::to_string(client->id)
                           + "\r\n"
                             "\r\n";

    TCPHandler::GetInstance().SendRTSPResponse(client->tcpSocket, response);

    mediaStreamHandler->SetCmd("TEARDOWN");
}
