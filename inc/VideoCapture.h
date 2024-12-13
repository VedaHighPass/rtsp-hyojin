#include <vector>
#include <mutex>
#include <utility>
#include <CUDAimageprocessing.h>


struct VCImage {
    unsigned char *img;
    unsigned int size;
    unsigned int timestamp;
};

class VideoCapture{
public:
    static const int buffer_max_size = 10;

    static VideoCapture& getInstance() {
        static VideoCapture instance;
        return instance;
    }
    inline bool isEmptyBuffer() { return (buffer_size == 0); };
    inline bool isFullBuffer() { return (buffer_size == buffer_max_size); };
    void pushImg(const VCImage& img);
    VCImage popImg();
private:
    std::vector <VCImage> imgBuffer;
    int head = 0;
    int tail = 0;
    int buffer_size = 0;

    std::mutex imgBufferMutex;

    VideoCapture()
    {
        //imgBuffer.resize(buffer_max_size, std::make_pair(new unsigned char[640*320], 640*320)); // 초기화
        imgBuffer.resize(buffer_max_size, {nullptr, 0, 0}); // 초기화
    }

    ~VideoCapture() {
        for (auto& pair : imgBuffer) {
            delete[] pair.img; // 동적 할당 해제
        }
    }
};
