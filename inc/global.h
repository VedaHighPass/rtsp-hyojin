#ifndef GLOBAL_H
#define GLOBAL_H
#include <string.h>


const int g_serverRtpPort = 554;


class ServerStream{
    public:
    static ServerStream& getInstance(){
        static ServerStream instance;
        return instance;
    }
};

static std::string g_inputFile = "example/dragon.h264";

#endif //GLOBAL_H
