#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"
#include "FrameBuffer.h"
#include "CUDAimageprocessing.h"
#include "VideoCapture.h"

// VIDEODEV "/dev/video0"
// WIDTH 640
// HEIGHT 360


class Camera {
public:
    Camera();
    ~Camera();

    void initFFmpeg(const char *filename);
    int get_fd() const;
    bool captureFrameBuffer(FrameBuffer& framebuffer);
    bool captureOpencv(Camera& camera);


private:
    struct Buffer {
        void* start;
        size_t length;
    };

    int fd, i = 0;
    int frameCounter = 0, image_count = 1;
    struct Buffer *buffers = NULL;
    volatile unsigned int n_buffers = 0;      /* 버퍼 개수 */

    AVCodecContext *codec_ctx;      // FFmpeg 코덱 컨텍스트
    AVFormatContext *fmt_ctx;       // FFmpeg 포맷 컨텍스트
    AVPacket *packet;               // FFmpeg 패킷 (인코딩된 데이터 저장)
    AVFrame *frame;                 // FFmpeg 프레임 (YUV420p 데이터 저장)
    int frame_index;                // 현재 프레임 인덱스



    // V4l2 Camera setup
    void initDevice();
    void initMMap();
    void startCapturing();
    void stopCapturing();

    // Frambuffer로 출력
    void processImage(const void* data, FrameBuffer& framebuffer);
    int clip(int value, int min, int max);


    // OpenCV로 출력
    void processRawImage(void* data, int width, int height);
    void apply_white_balance(cv::Mat& bayer_image);
    void apply_white_balance2(cv::Mat& bayer_image);



    // OpenCV & CUDA 출력
    void processRawImageCUDA(void* data, int width, int height);
    void applyWhiteBalance(cv::Mat& image);
    void applyGammaCorrection(cv::Mat& image, double gamma);



    // Test & Check
    void checkFormat();
    void saveFrameToFile(const void* data, size_t size);
    void processRGGBImage(void* data, int width, int height, const std::string& histogramCsvFilename,const std::string& histogramPngFilename);
    void processAndVisualizeRawData(void* data, int width, int height);
    void saveHistogramImage(uint8_t* data, int width, int height, const std::string& histogramImageFilename, const std::string& histogramCsvFilename);
    cv::Scalar computeSSIM(const cv::Mat& img1, const cv::Mat& img2);



    // FFmpeg
    void encodeFrame(const cv::Mat& cpuYUV420p, size_t size);


};

#endif // CAMERA_H
