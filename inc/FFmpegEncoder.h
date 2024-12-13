#ifndef FFMPEG_ENCODER_H
#define FFMPEG_ENCODER_H
#include <common.h>
#include "VideoCapture.h"
class AVStream;

class FFmpegEncoder {
public:
    FFmpegEncoder(const std::string& filename, int width, int height, double fps);
    ~FFmpegEncoder();
    void encode(const cv::Mat& frame, double fps);

private:
    void initFFmpeg(const std::string& filename, double fps);
    void releaseFFmpeg();

    // FFmpeg 관련 변수 선언
    struct AVCodecContext *codec_ctx = nullptr;
    struct AVFormatContext *fmt_ctx = nullptr;
    struct AVPacket *packet = nullptr;
    struct AVFrame *frame = nullptr;
    int frame_index = 0;
    AVStream *stream = nullptr;

    int width;
    int height;
};

#endif // FFMPEG_ENCODER_H

