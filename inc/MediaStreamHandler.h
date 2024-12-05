#ifndef RTSP_MEDIASTREAMHANDLER_H
#define RTSP_MEDIASTREAMHANDLER_H

#include <atomic>
#include <alsa/asoundlib.h>
#include <condition_variable>

enum MediaStreamState{
    eMediaStream_Init,
    eMediaStream_Play,
    eMediaStream_Pause,
    eMediaStream_Teardown,
};

class RtpPacket;
class MediaStreamHandler {
public:
    UDPHandler* udpHandler;

    MediaStreamHandler();

    void HandleMediaStream();

    void SetCmd(const std::string& cmd);

private:
    MediaStreamState streamState;
    std::mutex streamMutex;
    std::condition_variable condition; // condition variable for streaming state controll
    void SendFragmentedRTPPackets(unsigned char* payload, size_t payloadSize, RtpPacket& rtpPacket, const uint32_t timeStamp);
};

#endif //RTSP_MEDIASTREAMHANDLER_H
