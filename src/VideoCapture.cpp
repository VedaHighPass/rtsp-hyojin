#include "VideoCapture.h"
#include <cstring>
#include <iostream>


void VideoCapture::pushImg(unsigned char* imgPtr, int size) {
    if(bufferOpen & imgBuffer.size() < 10){
//        std::cout << "PUSH _ imgBufferSize = "<< imgBuffer.size() <<"("<< size<<")" <<std::endl;
        unsigned char* buffer = new unsigned char[size];
        memcpy(buffer, imgPtr, size);
        imgBufferMutex.lock();
        imgBuffer.push(std::make_pair(buffer, size));
        imgBufferMutex.unlock();

//        std::cout << "PUSH _ imgBufferSize = "<< imgBuffer.size() <<std::endl;
    }
}

std::pair<unsigned char*, int> VideoCapture::popImg() {
    std::pair<unsigned char*, int> ret = imgBuffer.front();
    std::cout << "POP _ imgBufferSize = "<< imgBuffer.size() << "(" << "," << ret.second <<")" <<std::endl;
    imgBufferMutex.lock();
    imgBuffer.pop();
    imgBufferMutex.unlock();
    return ret;
}
