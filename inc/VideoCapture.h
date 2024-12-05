#include <vector>
#include <mutex>
#include <utility>
#include <CUDAimageprocessing.h>

class VideoCapture {
public:
    bool bufferOpen = false;
    static const int buffer_max_size = 5; // 고정된 버퍼 크기

    static VideoCapture& getInstance() {
        static VideoCapture instance;
        return instance;
    }

    inline bool isEmptyBuffer() { return (buffer_size == 0); };
    inline bool isFullBuffer() { return (buffer_size == buffer_max_size); };

    void pushImg(unsigned char* imgPtr, int size);
    std::pair<unsigned char*, int> popImg();

private:
    std::vector<std::pair<unsigned char*, int>> imgBuffer; // 순환 큐용 버퍼
    int head = 0;   // 다음 pop 위치
    int tail = 0;   // 다음 push 위치
    int buffer_size = 0; // 현재 저장된 아이템 수

    std::mutex imgBufferMutex;

    VideoCapture() {
        imgBuffer.resize(buffer_max_size, std::make_pair(nullptr, 0)); // 초기화
    }

    ~VideoCapture() {
        for (auto& pair : imgBuffer) {
            delete[] pair.first; // 동적 할당 해제
        }
    }
};

