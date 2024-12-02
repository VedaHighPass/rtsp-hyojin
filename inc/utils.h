#ifndef RTSP_UTILS_H
#define RTSP_UTILS_H

#include <string>
#include <utility>
#include <cstdint>

class utils {
public:
    static uint64_t GetTime();
    static uint32_t GetRanNum(int n);
    static std::string GetIP();
};

#endif //RTSP_UTILS_H
