#ifndef RTSP_REQUESTHANDLER_H
#define RTSP_REQUESTHANDLER_H

#include <memory>
#include <string>
struct Info;
class MediaStreamHandler;


class RequestHandler {
public:
    RequestHandler(const std::shared_ptr<Info>& clit);

    void HandleRequest();

private:
    MediaStreamHandler *mediaStreamHandler;

    std::shared_ptr<Info> client;

    std::string ParseMethod(const std::string& request);

    int ParseCSeq(const std::string& request);

    std::pair<int, int> ParsePorts(const std::string& request);

    bool ParseAccept(const std::string& request);

    void HandleOptionsRequest(int cseq);

    void HandleDescribeRequest(const std::string& request, int cseq);

    void HandleSetupRequest(const std::string& request, int cseq);

    void HandlePlayRequest(int cseq);

    void HandlePauseRequest(int cseq);

    void HandleTeardownRequest(int cseq);
};

#endif //RTSP_REQUESTHANDLER_H