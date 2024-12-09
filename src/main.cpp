#include "common.h"
#include "CUDAimageprocessing.h"
#include "VideoCapture.h"
#include "ClientSession.h"
#include "TCPHandler.h"

// FFmpeg 관련 변수 선언
AVCodecContext *codec_ctx = nullptr;
AVFormatContext *fmt_ctx = nullptr;
AVPacket *packet = nullptr;
AVFrame *frame = nullptr;
int frame_index = 0;

void initFFmpeg(const char *filename, double fps) {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        throw std::runtime_error("Codec not found");
    }

    // 코덱 컨텍스트 할당
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    // 초당 처리되는 비트 양 설정(4,000,000 bits per second = 4000kbps)
    codec_ctx->bit_rate = 4000000;                  // 비트레이트
    codec_ctx->width = WIDTH_RESIZE;               // 인코딩할 이미지의 가로 해상도
    codec_ctx->height = HEIGHT_RESIZE;             // 인코딩할 이미지의 세로 해상도
    codec_ctx->time_base = (AVRational){1, 30}; // FPS = 10 (1/10초마다 프레임 생성)
    // 실제 FPS 기반으로 time_base 설정
//     codec_ctx->time_base = AVRational{1, static_cast<int>(fps)};
    codec_ctx->gop_size = 30;                      // GOP 크기 (10프레임마다 I-프레임)

    // I프레임과 P/B프레임간의 간격 설정 (GOP:Group of Pictures)
    // I프레임은 전체화면저장하는 완전한 프레임이고, P/B프레임은 이전 또는 다음프레임에 의존하는 차이프레임
    // gop_size = 10은 매 10프레임마다 I프레임을 삽입하는것
    codec_ctx->max_b_frames = 1;                   // 최대 B-프레임
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;       // 출력 포맷

    // 멀티스레드 활성화
    codec_ctx->thread_count = 4;
    codec_ctx->thread_type = FF_THREAD_FRAME;

    // 속도 우선 옵션
    av_opt_set(codec_ctx->priv_data, "preset", "ultrafast", 0);
    av_opt_set(codec_ctx->priv_data, "tune", "zerolatency", 0);

    // 코덱 열고 초기화
    if (avcodec_open2(codec_ctx, codec, NULL) < 0) {
        throw std::runtime_error("Could not open codec");
    }

    // ffmpeg에서 사용할 포멧 컨텍스트 할당
    fmt_ctx = avformat_alloc_context();
    if (!fmt_ctx) {
        throw std::runtime_error("Could not allocate format context");
    }

    // 출력 파일 포멧 결정(파일이름에 따라 추측)
    AVOutputFormat *output_fmt = av_guess_format(NULL, filename, NULL);
    fmt_ctx->oformat = output_fmt;

    //avio_open으로 파일 열기. 쓰기모드로 염
    //파일입출력을위한 I/O핸들러를 여는 함수. 주어진 파일을 쓰기모드로 열고, FFmpeg이 해당파일에 데이터 쓸 수 있게함
    // AVFormatContext구조체의 pb필드는 입출력 핸들임
    if (avio_open(&fmt_ctx->pb, filename, AVIO_FLAG_WRITE) < 0) {
        throw std::runtime_error("Could not open output file");
    }

    // ffmpeg라이브러리로 새 비디오 스트림 만들고 스트림 다양한 속성 설정 후 avformat_write_header로 파일헤더작성
    // 새 비디오 스트림 생성(avformat_new_stream함수는 AVFormatContext(fmt_ctx)에 새로운 AVStream (오디오 등등..)추가)
    // 두번째 인자는 사용할 코덱 명시 (NULL사용하여 기본코덱 사용하겠다는 의미)
    AVStream *stream = avformat_new_stream(fmt_ctx, NULL);
    stream->time_base = {1, 17};                   // 스트림 시간 기준
    stream->codecpar->codec_id = AV_CODEC_ID_H264; // H.264 코덱 설정
    stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO; // 미디어 타입 비디오로 설정(오디오도 있음)
    stream->codecpar->width = WIDTH_RESIZE;
    stream->codecpar->height = HEIGHT_RESIZE;
    stream->codecpar->format = AV_PIX_FMT_YUV420P;

    // 비디오 파일 헤더 작성
    int ret = avformat_write_header(fmt_ctx, NULL);
    if (ret < 0) {
      char err_buf[AV_ERROR_MAX_STRING_SIZE];
      av_strerror(ret, err_buf, sizeof(err_buf)); // 에러 메시지 변환
      throw std::runtime_error(std::string("Error writing header: ") + err_buf);
    }

    packet = av_packet_alloc();                    // 패킷 메모리 할당
    if (!packet) {
        throw std::runtime_error("Could not allocate av packet");
    }

    // 코덱 컨텍스트는 비디오나 오디오 데이터가 어떻게 압축되거나 풀리는지에 대한 정보 저장함.
    // H.264, AAC 등으로 압축할때 코덱 컨텍스트 설정하여 데이터 인코딩
    //
    // 포맷 컨텍스트는 비디오나 오디오 담고있는 컨테이너 포맷 처리시 사용. 컨테이너포맷은 비디오와 오디오 데이터 뿐만 아니라 자막,메타데이터 등등 다양한정보 함께 담고있음
    // MP4, MKV 등등의 정보와 데이터 관리

    frame = av_frame_alloc();                      // 프레임 메모리 할당
    frame->format = codec_ctx->pix_fmt;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    if (av_frame_get_buffer(frame, 32) < 0) {
        throw std::runtime_error("Could not allocate frame buffer");
    }

    frame_index = 0;                               // 초기 프레임 인덱스 설정
}


void encodeFrame(const cv::Mat& cpuYUV420p, double measuredFPS) {

    int y_size = codec_ctx->width * codec_ctx->height;      // Y 채널 크기
    int uv_size = y_size / 4;                              // U 및 V 채널 크기

    frame->data[0] = cpuYUV420p.data;                      // Y 채널
    frame->data[1] = cpuYUV420p.data + y_size;            // U 채널
    frame->data[2] = cpuYUV420p.data + y_size + uv_size;  // V 채널


    // PTS 설정 (프레임 인덱스를 사용하여 PTS 설정)
    // PTS는 각 프레임이 언제 표시되어야하는지를 나타내는 시간정보. 여기서 frame_index는 인코딩중인 프레임 순서 의미
    // PTS 계산 (이전 값보다 항상 증가 보장)
    static int64_t lastPTS = 0;
    int64_t currentPTS = static_cast<int64_t>(frame_index * codec_ctx->time_base.den / measuredFPS);

    // `PTS`가 이전 값보다 크도록 보장
    if (currentPTS <= lastPTS) {
        currentPTS = lastPTS + 1;
    }
    frame->pts = currentPTS;
    lastPTS = currentPTS; // PTS 업데이트

    frame_index++;


    // 프레임을 인코더로 보냄
    int ret = avcodec_send_frame(codec_ctx, frame);
    if ( ret < 0) {
        throw std::runtime_error("Error sending frame for encoding");
    }

    // 인코딩된 패킷을 가져오기
    while ( ret >=  0) {
        ret = avcodec_receive_packet(codec_ctx, packet);

        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
        if (ret < 0) {
            throw std::runtime_error("Error during encoding");
        }

        //인코딩된 패킷을 출력파일에 기록한 후 패킷을 해제함(unref)
        // av_interleaved_write_fraem 은 인코딩된 패킷을 출력파일에 기록하는 함수
        // 인터리빙 방식 : 오디오 및 비디오 스트림 교차저장하는 방식(영상재생시 동시에 재생되게함)
        // fmt_ctx(포맷컨텍스트) 출력파일과 관련된 정보 포함하고있으며, 파일에 패킷을 기록할 대상이 됨
        // pkt : 인코딩된 비디오또는 오디오 데이터 포함하고있음. 이 데이터를 출력파일에 기록하는것
//         av_interleaved_write_frame(fmt_ctx, packet);      // 패킷을 파일에 기록
        // 패킷에 할당된 메모리를 해제. 인코딩된 데이터를 파일에 기록한 후 메모리 해제해야함

        if (packet->size > 0 && packet->data) {
            VideoCapture::getInstance().pushImg(packet->data, packet->size);
        }

        av_packet_unref(packet);

    }

}


// FFmpeg 자원 해제 함수
void releaseFFmpeg() {
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (fmt_ctx) {
        avio_close(fmt_ctx->pb);
        avformat_free_context(fmt_ctx);
    }
}




int main() {

    // Replace with the correct GStreamer pipeline for your IMX219 camera
    const char *pipeline_str = "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1920,height=1080 ! nvvidconv ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
    cv::VideoCapture cap(pipeline_str, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open the GStreamer pipeline." << std::endl;
        return -1;
    }

    std::cout << "GStreamer pipeline initialized successfully." << std::endl;

    // Start RTSP Server
    std::thread([]() -> void
                {
        while(1){
            std::pair<int, std::string> newClient = TCPHandler::GetInstance().AcceptClientConnection();
            std::cout<<"Client connected" << std::endl;

            ClientSession* clientSession = new ClientSession(newClient);
            clientSession->StartRequestHandlerThread();
        } })
        .detach();

    int frameCount = 0;
    double fps = 30.0; // 초기 FPS 가정값
    double accumulatedFPS = 0.0;
    int fpsMeasurements = 0;
    auto start = std::chrono::high_resolution_clock::now();

    // FFmpeg 초기화
    initFFmpeg("output.h264", fps);

    try
    {
           for (int i = 0; i < 500; ++i) {

            cv::Mat frame;
            cap >> frame; // 프레임 읽기

            if (frame.empty()) {
                std::cerr << "Error: Received an empty frame." << std::endl;
                break;
            }

            frameCount++;
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();

            if (elapsed >= 1000) {
                double currentFPS = frameCount / (elapsed / 1000.0);
                accumulatedFPS += currentFPS;
                fpsMeasurements++;
                fps = accumulatedFPS / fpsMeasurements;

                std::cout << "Measured FPS: " << currentFPS << ", Average FPS: " << fps << std::endl;

                frameCount = 0;
                start = now;
            }

            cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 0, 255), 3);


            // CPU -> GPU
            cv::cuda::GpuMat gpuBGR, gpuYUV420p;
            gpuBGR.upload(frame);

            // BGR -.YUV420P
            BGRtoYUV420P(gpuBGR, gpuYUV420p);

            // GPU -> CPU
            cv::Mat cpuYUV420p;
            gpuYUV420p.download(cpuYUV420p);

            encodeFrame(cpuYUV420p, fps);

            // 화면에 프레임 표시
            cv::imshow("BGR Frame", frame);

            // 'q' 키를 누르면 종료
            if (cv::waitKey(1) == 'q') {
                break;
            }

        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
//
    return EXIT_SUCCESS;
}


