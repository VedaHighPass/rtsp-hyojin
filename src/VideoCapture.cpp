#include "VideoCapture.h"
#include <cstring>
#include <iostream>

void VideoCapture::pushImg(unsigned char* imgPtr, int size) {
    std::lock_guard<std::mutex> lock(imgBufferMutex);

    if (bufferOpen && !isFullBuffer()) {
        if (imgBuffer[tail].first == nullptr) {
            imgBuffer[tail].first = new unsigned char[size]; // 메모리 할당
        } else if (imgBuffer[tail].second < size) {
            delete[] imgBuffer[tail].first;
            imgBuffer[tail].first = new unsigned char[size]; // 재할당
        }

        memcpy(imgBuffer[tail].first, imgPtr, size);
        imgBuffer[tail].second = size;

        tail = (tail + 1) % buffer_max_size; // tail 위치 갱신
        buffer_size++;

        std::cout << "PUSH _ imgBufferSize = " << buffer_size << " (" << size << ")" << std::endl;
    }
}

std::pair<unsigned char*, int> VideoCapture::popImg() {
    std::lock_guard<std::mutex> lock(imgBufferMutex);

    if (isEmptyBuffer()) {
        std::cerr << "POP _ Buffer is empty!" << std::endl;
        return std::make_pair(nullptr, 0);
    }

    auto ret = imgBuffer[head];
    head = (head + 1) % buffer_max_size; // head 위치 갱신
    buffer_size--;

    std::cout << "POP _ imgBufferSize = " << buffer_size << " (" << ret.second << ")" << std::endl;

    return ret;
}

