#include "GStreamerCapture.h"
#include <iostream>

GStreamerCapture::GStreamerCapture(const std::string& pipeline) : cap(pipeline, cv::CAP_GSTREAMER) {
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Unable to open the GStreamer pipeline.");
    }
    std::cout << "GStreamer pipeline initialized successfully." << std::endl;
}

GStreamerCapture::~GStreamerCapture() {
    cap.release();
}

bool GStreamerCapture::read(cv::Mat& frame) {
    return cap.read(frame);
}

