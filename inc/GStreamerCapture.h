#ifndef GSTREAMER_CAPTURE_H
#define GSTREAMER_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>

class GStreamerCapture {
public:
    GStreamerCapture(const std::string& pipeline);
    ~GStreamerCapture();
    bool read(cv::Mat& frame);

private:
    cv::VideoCapture cap;
};

#endif // GSTREAMER_CAPTURE_H

