#include <queue>
#include <mutex>
#include <CUDAimageprocessing.h>

class VideoCapture{
public:
    bool bufferOpen = false;
    static VideoCapture& getInstance() {
        static VideoCapture instance;
        return instance;
    }
//    inline int GetBufferSize() { return imgBuffer.size(); };
    inline bool isEmptyBuffer() { return imgBuffer.empty(); };
    void pushImg(unsigned char* imgPtr, int size);
    std::pair<unsigned char*, int> popImg();

private:
    std::queue <std::pair<unsigned char*, int>> imgBuffer;
    std::mutex imgBufferMutex;
};
