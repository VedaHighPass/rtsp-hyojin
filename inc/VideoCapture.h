#include <queue>
#include <mutex>

class VideoCapture{
public:
    bool bufferOpen = false;
    static VideoCapture& getInstance() {
        static VideoCapture instance;
        return instance;
    }
    inline int getImgBufferSize() { return imgBuffer.size(); };
    void pushImg(unsigned char* imgPtr, int size);
    std::pair<unsigned char*, int> popImg();
private:
    std::queue <std::pair<unsigned char*, int>> imgBuffer;
    std::mutex imgBufferMutex;
};