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
    inline int GetBufferSize() { return imgBuffer.size(); };
    //void pushImg(unsigned char* imgPtr, int size);
    void pushImg(AVPacket* imgPtr);
    //std::pair<unsigned char*, int> popImg();
    AVPacket* popImg();

private:
    //std::queue <std::pair<unsigned char*, int>> imgBuffer;
    std::queue<AVPacket*> imgBuffer;
    std::mutex imgBufferMutex;
};
