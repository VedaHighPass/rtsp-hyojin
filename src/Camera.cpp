#include "common.h"
#include "Camera.h"

Camera::Camera() {
    fd = open(VIDEODEV, O_RDWR | O_NONBLOCK, 0);
    if (fd == -1) {
        throw std::runtime_error("Failed to open video device");
    }

    initDevice();
    initMMap();
    startCapturing();
}


Camera::~Camera() {
    stopCapturing();

        /* 메모리 해제 */
    for (int i = 0; i < n_buffers; ++i)
    {
        if (-1 == munmap(buffers[i].start, buffers[i].length));
    }

    free(buffers);

    // FFmpeg 종료 작업
    av_write_trailer(fmt_ctx);  // 트레일러 작성
    avcodec_close(codec_ctx);   // 코덱 닫기
    avio_close(fmt_ctx->pb);    // 파일 닫기
    avformat_free_context(fmt_ctx); // 포맷 컨텍스트 해제
    av_packet_free(&packet);

    close(fd);
}


int Camera::get_fd() const {
    return fd;
}


void Camera::initFFmpeg(const char *filename) {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        throw std::runtime_error("Codec not found");
    }

    // 코덱 컨텍스트 할당
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        throw std::runtime_error("Could not allocate codec context");
    }

    // 초당 처리되는 비트 양 설정(400,000 bits per second = 400kbps)
    codec_ctx->bit_rate = 400000;                  // 비트레이트
    codec_ctx->width = WIDTH_RESIZE;               // 인코딩할 이미지의 가로 해상도
    codec_ctx->height = HEIGHT_RESIZE;             // 인코딩할 이미지의 세로 해상도
//     codec_ctx->time_base = {1, 25};                // 25fps 설정(1/25)
    codec_ctx->time_base = {1,10};                 // 10fps 설정(1000ms/100ms)
    codec_ctx->gop_size = 10;                      // GOP 크기 (10프레임마다 I-프레임)

    // I프레임과 P/B프레임간의 간격 설정 (GOP:Group of Pictures)
    // I프레임은 전체화면저장하는 완전한 프레임이고, P/B프레임은 이전 또는 다음프레임에 의존하는 차이프레임
    // gop_size = 10은 매 10프레임마다 I프레임을 삽입하는것
    codec_ctx->max_b_frames = 1;                   // 최대 B-프레임
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;       // 출력 포맷

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
    stream->time_base = {1, 25};                   // 스트림 시간 기준
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



void Camera::initDevice() {

    struct v4l2_capability cap{};
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        throw std::runtime_error("Failed to query V4L2 device capabilities");
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) || !(cap.capabilities & V4L2_CAP_STREAMING)) {
        throw std::runtime_error("Device does not support required capabilities");
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        throw std::runtime_error("Device does not support streaming");
    }



    struct v4l2_cropcap cropcap{};
    struct v4l2_crop crop{};

//     /* 비디오 입력 및 크롭 설정 */
//     memset(&cropcap, 0, sizeof(cropcap));  /* cropcap 구조체의 모든 필드를 0으로 초기화 */
//     cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  /* 크롭 설정을 위한 버퍼 타입을 비디오 캡처로 설정 */
//     if (0 == ioctl(fd, VIDIOC_CROPCAP, &cropcap)) {  /* VIDIOC_CROPCAP: 장치의 크롭 기능 지원 여부 확인 */
//         crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;      /* 크롭 버퍼 타입을 비디오 캡처로 설정 */
//         crop.c = cropcap.defrect;  /* 기본 크롭 영역을 장치의 기본값으로 설정 (defrect는 장치의 기본 크롭 영역) */
//           /* VIDIOC_S_CROP: 크롭 영역을 설정 (드라이버에 따라 지원되지 않을 수 있음) */
//         if (ioctl(fd, VIDIOC_S_CROP, &crop) == -1) {
//           throw std::runtime_error("crop");
//         }
//     }
//

//     struct v4l2_format fmt{};
//     memset(&fmt, 0, sizeof(fmt));
//     fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
//     fmt.fmt.pix.width = WIDTH;
//     fmt.fmt.pix.height = HEIGHT;
//     //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
//     fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10;
//     fmt.fmt.pix.field = V4L2_FIELD_NONE;
//
//     if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
//         throw std::runtime_error("Failed to set format");
//     }
}

void Camera::initMMap() {
    struct v4l2_requestbuffers req{};
    memset(&req, 0, sizeof(req));  /* req 구조체의 모든 필드를 0으로 초기화 */
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        throw std::runtime_error("Failed to request buffers");
    }

    if (req.count < 2) {
        throw std::runtime_error("Insufficient buffer memory");
    }


    /* 요청한 버퍼 수만큼의 메모리를 할당 (calloc을 사용해 메모리를 0으로 초기화) */
    buffers = static_cast<Buffer*>(calloc(req.count, sizeof(*buffers)));  /* 버퍼 정보를 저장할 메모리를 동적으로 할당 */
    if (!buffers) {
        throw std::runtime_error("Out of memory");
    }

        /* 각 버퍼에 대해 메모리 맵핑 */
    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));  /* buf 구조체 초기화 */
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;  /* 비디오 캡처 타입으로 버퍼 타입 설정 */
        buf.memory = V4L2_MEMORY_MMAP;  /* 메모리 매핑 방식 설정 */
        buf.index = n_buffers;  /* 버퍼의 인덱스 설정 (0부터 시작) */

        /* 버퍼 정보를 조회 (VIDIOC_QUERYBUF 호출) */
        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
            throw std::runtime_error("VIDIOC_QUERYBUF");
        }


        buffers[n_buffers].length = buf.length;  /* 각 버퍼의 길이를 저장 (버퍼의 크기) */
        buffers[n_buffers].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                                        MAP_SHARED, fd, buf.m.offset);  /* 메모리 매핑 수행 */
        if (MAP_FAILED == buffers[n_buffers].start)
            throw std::runtime_error("mmap");
    }

}

void Camera::startCapturing() {

    for (int i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        /* 버퍼를 큐에 넣음 */
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            throw std::runtime_error("VIDIOC_QBUF");
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        throw std::runtime_error("Failed to start streaming");
    }
}

void Camera::stopCapturing() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        throw std::runtime_error("Failed to stop streaming");
    }

}








//---------------------------------Using-Framebuffer-output-start------------------------------------//

int Camera::clip(int value, int min, int max) {
    return (value > max ? max : value < min ? min : value);
}


void Camera::processImage(const void* data, FrameBuffer& framebuffer) {
    const uint16_t* in = static_cast<const uint16_t*>(data); // RG10 Bayer 데이터
    int fb_width = framebuffer.getScreenInfo().xres;        // 프레임버퍼 가로 크기
    int fb_height = framebuffer.getScreenInfo().yres;       // 프레임버퍼 세로 크기
    int depth_fb = framebuffer.getScreenInfo().bits_per_pixel / 8;
    size_t location = 0; // 프레임버퍼에서 현재 위치를 추적할 변수
    uint8_t r, g, b, a = 0xff; // RGBA의 알파 채널(A)는 고정값 255

    // Bayer 데이터의 한 행의 데이터 크기 (픽셀당 2바이트)
    int istride = WIDTH;

    // 프레임버퍼 크기만큼 데이터 처리
    for (int y_fb = 0; y_fb < fb_height; ++y_fb) {
        for (int x_fb = 0; x_fb < fb_width; ++x_fb) {
            int x = x_fb; // 입력 영상의 x 좌표 (잘라내기 없음)
            int y = y_fb; // 입력 영상의 y 좌표 (잘라내기 없음)

            // Bayer 패턴에 따른 R, G, B 계산
            if ((y % 2 == 0) && (x % 2 == 0)) { // 빨강 픽셀
                r = in[y * istride + x] >> 2;
                g = ((x > 0 ? in[y * istride + x - 1] : in[y * istride + x]) +
                     (x < istride - 1 ? in[y * istride + x + 1] : in[y * istride + x])) >> 3;
                b = (y < HEIGHT - 1 ? in[(y + 1) * istride + x] : in[y * istride + x]) >> 2;
            } else if ((y % 2 == 0) && (x % 2 == 1)) { // 초록 픽셀 (짝수 행, 홀수 열)
                g = in[y * istride + x] >> 2;
                r = ((x > 0 ? in[y * istride + x - 1] : in[y * istride + x]) +
                     (x < istride - 1 ? in[y * istride + x + 1] : in[y * istride + x])) >> 3;
                b = ((y < HEIGHT - 1 ? in[(y + 1) * istride + x - 1] : in[y * istride + x]) +
                     (y < HEIGHT - 1 ? in[(y + 1) * istride + x + 1] : in[y * istride + x])) >> 3;
            } else if ((y % 2 == 1) && (x % 2 == 0)) { // 초록 픽셀 (홀수 행, 짝수 열)
                g = in[y * istride + x] >> 2;
                b = ((x > 0 ? in[y * istride + x - 1] : in[y * istride + x]) +
                     (x < istride - 1 ? in[y * istride + x + 1] : in[y * istride + x])) >> 3;
                r = ((y > 0 ? in[(y - 1) * istride + x] : in[y * istride + x]) +
                     (y < HEIGHT - 1 ? in[(y + 1) * istride + x] : in[y * istride + x])) >> 3;
            } else { // 파랑 픽셀
                b = in[y * istride + x] >> 2;
                g = ((x > 0 ? in[y * istride + x - 1] : in[y * istride + x]) +
                     (x < istride - 1 ? in[y * istride + x + 1] : in[y * istride + x])) >> 3;
                r = (y > 0 ? in[(y - 1) * istride + x] : in[y * istride + x]) >> 2;
            }

            // BGRA로 변환하여 프레임버퍼에 저장
            *(framebuffer.get_fbp() + location++) = b; // 파랑
            *(framebuffer.get_fbp() + location++) = g; // 초록
            *(framebuffer.get_fbp() + location++) = r; // 빨강
            *(framebuffer.get_fbp() + location++) = a; // 알파
        }
    }
}


bool Camera::captureFrameBuffer(FrameBuffer& framebuffer) {
    struct v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN) return false;
        throw std::runtime_error("Failed to dequeue buffer");
    }

    processImage(buffers[buf.index].start, framebuffer);


    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        throw std::runtime_error("Failed to queue buffer");
    }
    return true;
}

//-------------------------------------Using-Framebuffer-output-end-------------------------------------//








//-----------------------------------Test&Check Funcion start-------------------------------------------//


void Camera::processAndVisualizeRawData(void* data, int width, int height) {
    int totalPixels = width * height;
    auto* rawData = reinterpret_cast<uint8_t*>(data);

    // 1. 홀수 행 (RGRG...) 데이터 분석
    printf("\n--- RAW DATA ANALYSIS (ODD ROWS) ---\n");
    for (int row = 1232; row < std::min(1232, height); row += 2) { // 홀수 행만 처리
        printf("Row %d:\n", row);
       for (int col = 1590; col < std::min(1690, width); col++) { // 첫 10픽셀만 출력
            size_t offset = (row * width + col) * 2;
            uint8_t low = rawData[offset];       // 하위 바이트
            uint8_t high = rawData[offset + 1]; // 상위 바이트
            uint16_t value = (high << 8 | low);  // 16비트 값
            uint16_t lower10 = value & 0x03FF;  // 하위 10비트 추출
            uint16_t upper6 = (value >> 10) & 0x003F; // 상위 6비트 추출

            printf("Pixel[%d]: Low = 0x%02X, High = 0x%02X, 16-bit Value = 0x%04X, Lower 10 = %d, Upper 6 = %d\n",
                   col, low, high, value, lower10, upper6);
        }
    }

    // 2. 짝수 행 (GBGB...) 데이터 분석
    printf("\n--- RAW DATA ANALYSIS (EVEN ROWS) ---\n");
    for (int row = 1232; row < std::min(1233, height); row += 2) { // 짝수 행만 처리
        printf("Row %d:\n", row);
        for (int col = 1590; col < std::min(1690, width); col++) { // 첫 10픽셀만 출력
            size_t offset = (row * width + col) * 2;
            uint8_t low = rawData[offset];       // 하위 바이트
            uint8_t high = rawData[offset + 1]; // 상위 바이트
            uint16_t value = (high << 8 | low);  // 16비트 값
            uint16_t lower10 = value & 0x03FF;  // 하위 10비트 추출
            uint16_t upper6 = (value >> 10) & 0x003F; // 상위 6비트 추출

            printf("Pixel[%d]: Low = 0x%02X, High = 0x%02X, 16-bit Value = 0x%04X, Lower 10 = %d, Upper 6 = %d\n",
                   col, low, high, value, lower10, upper6);
        }
    }
}



void Camera::processRGGBImage(void* data, int width, int height, const std::string& histogramCsvFilename, const std::string& histogramPngFilename) {
    // RGGB 데이터 크기
    size_t dataSize = width * height * 2; // 2 bytes per pixel
    auto* rawData = reinterpret_cast<uint8_t*>(data);

    // 유효성 확인
    if (rawData == nullptr) {
        throw std::runtime_error("Input data is null");
    }

    // 히스토그램 초기화
    std::vector<int> histogramR(1024, 0); // R 값 히스토그램
    std::vector<int> histogramG(1024, 0); // G 값 히스토그램
    std::vector<int> histogramB(1024, 0); // B 값 히스토그램

    // RGGB 패턴 데이터 처리
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; col += 2) { // 2픽셀씩 처리
            size_t offset = (row * width + col) * 2; // 현재 픽셀의 시작 위치 (byte 단위)
            if (offset >= dataSize) {
                throw std::runtime_error("Buffer access out of bounds");
            }

            uint16_t low1 = rawData[offset];
            uint16_t high1 = rawData[offset + 1];
            uint16_t low2 = rawData[offset + 2];
            uint16_t high2 = rawData[offset + 3];

            if (row % 2 == 0) { // 짝수 행: GRGR
                uint16_t g1 = (high1 << 8 | low1) & 0x03FF; // G 값 1
                uint16_t r = (high2 << 8 | low2) & 0x03FF;  // R 값
                histogramG[g1]++;
                histogramR[r]++;
            } else { // 홀수 행: BGBG
                uint16_t b = (high1 << 8 | low1) & 0x03FF;  // B 값
                uint16_t g2 = (high2 << 8 | low2) & 0x03FF; // G 값 2
                histogramB[b]++;
                histogramG[g2]++;
            }
        }
    }

    // 히스토그램 CSV 파일로 저장
    std::ofstream csvFile(histogramCsvFilename);
    if (!csvFile.is_open()) {
        throw std::runtime_error("Failed to open CSV file for writing");
    }

    csvFile << "Value,Count_R,Count_G,Count_B\n";
    for (int i = 0; i < 1024; ++i) {
        csvFile << i << "," << histogramR[i] << "," << histogramG[i] << "," << histogramB[i] << "\n";
    }
    csvFile.close();
    std::cout << "Histogram data saved to: " << histogramCsvFilename << std::endl;

    // 히스토그램을 이미지로 시각화
    const int histWidth = 1024;       // 히스토그램 이미지의 너비
    const int histHeight = 400;      // 히스토그램 이미지의 높이
    const int binWidth = histWidth / 1024;

    // OpenCV 히스토그램 이미지 생성
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    // 히스토그램 데이터 정규화
    int maxR = *std::max_element(histogramR.begin(), histogramR.end());
    int maxG = *std::max_element(histogramG.begin(), histogramG.end());
    int maxB = *std::max_element(histogramB.begin(), histogramB.end());

    for (int i = 0; i < 1024; ++i) {
        int rHeight = static_cast<int>(static_cast<double>(histogramR[i]) / maxR * histHeight);
        int gHeight = static_cast<int>(static_cast<double>(histogramG[i]) / maxG * histHeight);
        int bHeight = static_cast<int>(static_cast<double>(histogramB[i]) / maxB * histHeight);

        // R 채널: 빨강
        cv::line(histImage, cv::Point(i * binWidth, histHeight),
                 cv::Point(i * binWidth, histHeight - rHeight), cv::Scalar(0, 0, 255), 1);

        // G 채널: 초록
        cv::line(histImage, cv::Point(i * binWidth, histHeight),
                 cv::Point(i * binWidth, histHeight - gHeight), cv::Scalar(0, 255, 0), 1);

        // B 채널: 파랑
        cv::line(histImage, cv::Point(i * binWidth, histHeight),
                 cv::Point(i * binWidth, histHeight - bHeight), cv::Scalar(255, 0, 0), 1);
    }

printf("File: %s | Line: %d | Function: %s | Message: %s\033[0m\n", __FILE__, __LINE__, __FUNCTION__,"ohh" );
    // 히스토그램 이미지를 파일로 저장
    if (!cv::imwrite(histogramPngFilename, histImage)) {
        throw std::runtime_error("Failed to save histogram PNG");
    }
printf("File: %s | Line: %d | Function: %s | Message: %s\033[0m\n", __FILE__, __LINE__, __FUNCTION__,"ohh" );

    std::cout << "Histogram image saved to: " << histogramPngFilename << std::endl;
}




void Camera::checkFormat() {
    struct v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (ioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
        throw std::runtime_error("Failed to get format");
    }

    // 포맷 정보 출력
    printf("Width: %d\n", fmt.fmt.pix.width);
    printf("Height: %d\n", fmt.fmt.pix.height);
    printf("Pixel Format: %c%c%c%c\n",
           fmt.fmt.pix.pixelformat & 0xFF,
           (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
           (fmt.fmt.pix.pixelformat >> 16) & 0xFF,
           (fmt.fmt.pix.pixelformat >> 24) & 0xFF);
    printf("Bytes per Line: %d\n", fmt.fmt.pix.bytesperline);
    printf("Size Image: %d\n", fmt.fmt.pix.sizeimage);
}


void Camera::saveFrameToFile(const void* data, size_t size) {

    // 현재 시간(Timestamp)을 가져옴
    std::time_t now = std::time(nullptr);
    struct tm* timeInfo = std::localtime(&now);

    // 파일 이름 생성: frame_0001_YYYYMMDD_HHMMSS.raw
    std::ostringstream filename;
    filename << "frame_"
             << std::setw(4) << std::setfill('0') << frameCounter++ << "_"
             << std::put_time(timeInfo, "%Y%m%d_%H%M%S") << ".raw";

    // 파일에 데이터 저장
    std::ofstream outfile(filename.str(), std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename.str());
    }

    outfile.write(reinterpret_cast<const char*>(data), size);
    if (!outfile) {
        throw std::runtime_error("Failed to write data to file: " + filename.str());
    }

    outfile.close();
    std::cout << "Saved frame to " << filename.str() << std::endl;
}


// SSIM 계산 함수
cv::Scalar Camera::computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;

    cv::Mat img1_32f, img2_32f;
    img1.convertTo(img1_32f, CV_32F);
    img2.convertTo(img2_32f, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_32f, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_32f, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_mu2 = mu1.mul(mu2);
    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);

    cv::Mat sigma1, sigma2;
    cv::GaussianBlur(img1_32f.mul(img1_32f), sigma1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_32f.mul(img2_32f), sigma2, cv::Size(11, 11), 1.5);

    sigma1 -= mu1_sq;
    sigma2 -= mu2_sq;

    cv::Mat sigma12;
    cv::GaussianBlur(img1_32f.mul(img2_32f), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1 = 2 * mu1_mu2 + C1;
    cv::Mat t2 = 2 * sigma12 + C2;
    cv::Mat t3 = t1.mul(t2);

    t1 = mu1_sq + mu2_sq + C1;
    t2 = sigma1 + sigma2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map;
    divide(t3, t1, ssim_map);
    cv::Scalar mssim = mean(ssim_map);
    return mssim;
}



void Camera::saveHistogramImage(uint8_t* data, int width, int height, const std::string& histogramImageFilename, const std::string& histogramCsvFilename) {

  int totalPixels = width * height;

    // 히스토그램 초기화
    std::vector<int> histogram(65536, 0);

    // 데이터 처리: 8비트 데이터를 두 개씩 합쳐 16비트 생성
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            size_t offset = (row * width + col) * 2; // 두 개의 8비트 데이터를 하나의 16비트로 결합
            uint8_t low = data[offset];             // 하위 바이트
            uint8_t high = data[offset + 1];        // 상위 바이트

            uint16_t value = (high << 8) | low;     // 16비트 값 생성
            histogram[value]++;                     // 히스토그램 업데이트
        }
    }


    // 1. CSV 파일로 히스토그램 데이터 저장
    std::ofstream csvFile(histogramCsvFilename);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Unable to open CSV file: " << histogramCsvFilename << std::endl;
        return;
    }

    csvFile << "Value,Count\n";
    for (int i = 0; i < histogram.size(); ++i) {
        if (histogram[i] > 0) {
            csvFile << i << "," << histogram[i] << "\n";
        }
    }
    csvFile.close();
    std::cout << "Histogram data saved to: " << histogramCsvFilename << std::endl;

    // 2. 히스토그램 시각화를 위한 이미지 생성
    int histWidth = 1024; // 히스토그램 이미지 너비
    int histHeight = 400; // 히스토그램 이미지 높이
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(0, 0, 0));

    // 히스토그램 데이터 정규화
    int maxCount = *std::max_element(histogram.begin(), histogram.end());
    std::vector<int> histDisplay(histWidth, 0);
    for (int i = 0; i < histWidth; ++i) {
        int binStart = i * (65536 / histWidth);
        int binEnd = (i + 1) * (65536 / histWidth);
        for (int j = binStart; j < binEnd; ++j) {
            histDisplay[i] += histogram[j];
        }
        histDisplay[i] = static_cast<int>(static_cast<double>(histDisplay[i]) / maxCount * histHeight);
    }

    // 히스토그램 이미지 생성
    for (int i = 0; i < histWidth; ++i) {
        cv::line(histImage,
                 cv::Point(i, histHeight),
                 cv::Point(i, histHeight - histDisplay[i]),
                 cv::Scalar(255, 255, 255), // 흰색 선으로 히스토그램 표시
                 1, cv::LINE_8);
    }

    // 히스토그램 이미지를 파일로 저장
    if (!cv::imwrite(histogramImageFilename, histImage)) {
        std::cerr << "Error: Unable to save histogram image: " << histogramImageFilename << std::endl;
        return;
    }

    std::cout << "Histogram image saved to: " << histogramImageFilename << std::endl;
}

//-----------------------------------Test&Check Funcion end-------------------------------------------//
















//-------------------------------------Using-OpenCV-output-start-------------------------------------//


// RAW 파일을 처리하여 RGB 이미지로 변환하는 함수
void Camera::processRawImage(void* data, int width, int height) {

    uint16_t* raw = static_cast<uint16_t*>(data); // RG10 Bayer 데이터

    // 화이트 밸런스 적용(CV_16U1)
    //apply_white_balance(raw);
    //apply_white_balance2(raw);


    // Bayer 데이터를 OpenCV Mat로 변환
    cv::Mat bayer_image_16bit(height, width, CV_16UC1, reinterpret_cast<char*>(const_cast<uint16_t*>(raw)));

    // 이미지 정보 출력
//     std::cout << "Image size: " << bayer_image_16bit.cols << "x" << bayer_image_16bit.rows << std::endl;
//     std::cout << "Image type 0(CV_8U), 1(CV_8S), 2(CV_16U): " << bayer_image_16bit.depth() << std::endl;
//     std::cout << "Number of channels(C1,C3): " << bayer_image_16bit.channels() << std::endl;
//     double min_val, max_val;
//     cv::minMaxLoc(bayer_image_16bit, &min_val, &max_val);
//     std::cout << "Min pixel value: " << min_val << ", Max pixel value: " << max_val << std::endl;

    // 정규화 수행 (CV_16UC1 → CV_8UC1)
    cv::Mat normalized_image;
    bayer_image_16bit.convertTo(normalized_image, CV_8UC1, 255.0 / 65535.0);
//     std::cout << "Nomalized Image size: " << normalized_image.cols << "x" << normalized_image.rows << std::endl;
//     std::cout << "INomalized Image depth: " << normalized_image.depth() << std::endl;
//     std::cout << "NNomalized Number of channels: " << normalized_image.channels() << std::endl;
//     cv::minMaxLoc(normalized_image, &min_val, &max_val);
//     std::cout << "Nomalized Min pixel value: " << min_val << ", Nomalized Max pixel value: " << max_val << std::endl;


    // Bayer(CV_8UC1) -> RGB 변환 (CV_8UC3) 디바이커링
    cv::Mat rgb_image, rgb_image_vng, rgb_image_ea;
//     cv::cvtColor(normalized_image, rgb_image, cv::COLOR_BayerRG2RGB);  // 디마이커링
//     cv::cvtColor(normalized_image, rgb_image_vng, cv::COLOR_BayerRG2RGB_VNG);  // 디마이커링
//     cv::cvtColor(normalized_image, rgb_image_ea, cv::COLOR_BayerRG2RGB_EA);  // 디마이커링
    cv::cvtColor(normalized_image, rgb_image_ea, cv::COLOR_BayerRG2RGB_EA);  // 디마이커링


//     // 품질 평가
//     double psnr = cv::PSNR(rgb_image_vng,rgb_image);
//     std::cout << "PSNR(vng,rgb): " << psnr << std::endl;
//
//     psnr = cv::PSNR(rgb_image_vng,rgb_image_ea);
//     std::cout << "PSNR(vng,ea): " << psnr << std::endl;
//
//     psnr = cv::PSNR(rgb_image_ea,rgb_image);
//     std::cout << "PSNR(ea,rgb): " << psnr << std::endl;
//
//     // SSIM 계산
//     cv::Scalar ssim = computeSSIM(rgb_image, rgb_image_vng);
//     std::cout << "SSIM(rgb,vng): " << ssim[0] << std::endl;
//
//     ssim = computeSSIM(rgb_image, rgb_image_ea);
//     std::cout << "SSIM(rgb,ea): " << ssim[0] << std::endl;
//
//     ssim = computeSSIM(rgb_image_ea, rgb_image_vng);
//     std::cout << "SSIM(ea,vng): " << ssim[0] << std::endl;
//

//     // 이미지 표시 및 저장
//     cv::imshow(" normalized Image", normalized_image);
//     cv::imwrite("Normalized_Image.png", normalized_image);
//
//     cv::waitKey(0);
//
//     cv::imshow(" rgb Image", rgb_image);
//     cv::imwrite("rgb_Image.png", rgb_image);
//
//     cv::waitKey(0);
//
//
//     cv::imshow("vng Image", rgb_image_vng);
//     cv::imwrite("vng_Image.png", rgb_image_vng);
//
//     cv::waitKey(0);
//

    cv::Mat resizedImage;
    cv::resize(rgb_image_ea, resizedImage, cv::Size(1280,960));
    cv::imshow("ea Image", resizedImage);

    if(cv::waitKey(1) >= 0)
    {
      throw std::runtime_error("Quit");
    }
//     cv::imwrite("EA_Image.png", rgb_image_ea);


//     cv::waitKey(0);


}

void Camera::apply_white_balance(cv::Mat& bayer_image) {
    // Bayer 이미지에서 각 채널의 평균 계산
    int width = bayer_image.cols;
    int height = bayer_image.rows;

    double sum_r = 0, sum_g = 0, sum_b = 0;
    int count_r = 0, count_g = 0, count_b = 0;

    // Bayer RGGB 패턴 기반으로 각 채널의 평균 계산
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t pixel = bayer_image.at<uint16_t>(y, x);
            if (y % 2 == 0 && x % 2 == 0) {  // Red 채널
                sum_r += pixel;
                count_r++;
            } else if (y % 2 == 0 && x % 2 == 1) {  // Green 채널 (Red Row)
                sum_g += pixel;
                count_g++;
            } else if (y % 2 == 1 && x % 2 == 0) {  // Green 채널 (Blue Row)
                sum_g += pixel;
                count_g++;
            } else if (y % 2 == 1 && x % 2 == 1) {  // Blue 채널
                sum_b += pixel;
                count_b++;
            }
        }
    }

    // 평균 값 계산
    double avg_r = sum_r / count_r;
    double avg_g = sum_g / count_g;
    double avg_b = sum_b / count_b;

    // 게인 계산 (Green을 기준으로 정규화)
    double gain_r = avg_g / avg_r;
    double gain_b = avg_g / avg_b;

    std::cout << "White balance gains: R=" << gain_r << ", G=1.0, B=" << gain_b << std::endl;

    // 각 채널에 게인 적용
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint16_t& pixel = bayer_image.at<uint16_t>(y, x);
            if (y % 2 == 0 && x % 2 == 0) {  // Red 채널
                pixel = cv::saturate_cast<uint16_t>(pixel * gain_r);
            } else if (y % 2 == 1 && x % 2 == 1) {  // Blue 채널
                pixel = cv::saturate_cast<uint16_t>(pixel * gain_b);
            }
            // Green 채널은 그대로 유지
        }
    }
}

void Camera::apply_white_balance2(cv::Mat& bayer_image) {
    int width = bayer_image.cols;
    int height = bayer_image.rows;

    // 1. Bayer 이미지에서 R, G, B 채널 합계와 픽셀 개수 계산
    double sum_r = 0, sum_g1 = 0, sum_g2 = 0, sum_b = 0;
    int count_r = 0, count_g1 = 0, count_g2 = 0, count_b = 0;

    // OpenCV 병렬 처리로 채널별 합계와 픽셀 개수를 계산
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        double local_sum_r = 0, local_sum_g1 = 0, local_sum_g2 = 0, local_sum_b = 0;
        int local_count_r = 0, local_count_g1 = 0, local_count_g2 = 0, local_count_b = 0;

        for (int y = range.start; y < range.end; ++y) {
            for (int x = 0; x < width; ++x) {
                uint16_t pixel = bayer_image.at<uint16_t>(y, x);
                if (y % 2 == 0 && x % 2 == 0) {  // Red 채널
                    local_sum_r += pixel;
                    local_count_r++;
                } else if (y % 2 == 0 && x % 2 == 1) {  // Green 채널 (G1)
                    local_sum_g1 += pixel;
                    local_count_g1++;
                } else if (y % 2 == 1 && x % 2 == 0) {  // Green 채널 (G2)
                    local_sum_g2 += pixel;
                    local_count_g2++;
                } else if (y % 2 == 1 && x % 2 == 1) {  // Blue 채널
                    local_sum_b += pixel;
                    local_count_b++;
                }
            }
        }

        // OpenCV 병렬 연산에 따른 지역 변수 합산
        cv::parallel_for_(cv::Range(0, 1), [&](const cv::Range&) {
            sum_r += local_sum_r;
            count_r += local_count_r;

            sum_g1 += local_sum_g1;
            count_g1 += local_count_g1;

            sum_g2 += local_sum_g2;
            count_g2 += local_count_g2;

            sum_b += local_sum_b;
            count_b += local_count_b;
        });
    });

    // G1, G2 평균을 통합하여 Green 채널 평균 계산
    double avg_r = sum_r / count_r;
    double avg_g = (sum_g1 / count_g1 + sum_g2 / count_g2) / 2.0;
    double avg_b = sum_b / count_b;

    // R, B 게인 계산
    double gain_r = avg_g / avg_r;
    double gain_b = avg_g / avg_b;

    // 2. LUT 생성 (0~65535 범위)
    std::vector<uint16_t> lut_r(65536), lut_b(65536);
    cv::parallel_for_(cv::Range(0, 65536), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            lut_r[i] = cv::saturate_cast<uint16_t>(i * gain_r);
            lut_b[i] = cv::saturate_cast<uint16_t>(i * gain_b);
        }
    });

    // 3. LUT를 사용해 화이트 밸런스 적용
    cv::parallel_for_(cv::Range(0, height), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            for (int x = 0; x < width; ++x) {
                uint16_t& pixel = bayer_image.at<uint16_t>(y, x);
                if (y % 2 == 0 && x % 2 == 0) {  // Red 채널
                    pixel = lut_r[pixel];
                } else if (y % 2 == 1 && x % 2 == 1) {  // Blue 채널
                    pixel = lut_b[pixel];
                }
                // Green 채널은 LUT 적용하지 않음
            }
        }
    });
}



//-------------------------------------Using-OpenCV-output-end-------------------------------------//


















//------------------------------------Using-OpenCV-CUDA-output-start-------------------------------------//


void Camera::encodeFrame(const cv::Mat& cpuYUV420p, size_t size) {

    int y_size = codec_ctx->width * codec_ctx->height;      // Y 채널 크기
    int uv_size = y_size / 4;                              // U 및 V 채널 크기

    // Y, U, V 채널 데이터를 AVFrame에 복사
    memcpy(frame->data[0], cpuYUV420p.data, y_size);               // Y 채널
    memcpy(frame->data[1], cpuYUV420p.data + y_size, uv_size);     // U 채널
    memcpy(frame->data[2], cpuYUV420p.data + y_size + uv_size, uv_size); // V 채널

//     for( ; i < 1 ; i++)
//     {
//       printf("FFmpeg linesize[0] (Y): %d\n", frame->linesize[0]);
//       printf("FFmpeg linesize[1] (U): %d\n", frame->linesize[1]);
//       printf("FFmpeg linesize[2] (V): %d\n", frame->linesize[2]);
//
//     }
// //FFmpeg의 linesize > OpenCV의 step
// int y_stride = codec_ctx->width;          // OpenCV에서 한 행의 stride (Y 채널: 1920)
// int uv_stride = codec_ctx->width / 2;     // U/V 채널: 960
//
// // Y 채널 복사
// for (int i = 0; i < codec_ctx->height; ++i) {
//     memcpy(frame->data[0] + i * frame->linesize[0],
//            cpuYUV420p.data + i * y_stride,
//            y_stride);
// }
//
// // U 채널 복사
// for (int i = 0; i < codec_ctx->height / 2; ++i) {
//     memcpy(frame->data[1] + i * frame->linesize[1],
//            cpuYUV420p.data + y_size + i * uv_stride,
//            uv_stride);
// }
//
// // V 채널 복사
// for (int i = 0; i < codec_ctx->height / 2; ++i) {
//     memcpy(frame->data[2] + i * frame->linesize[2],
//            cpuYUV420p.data + y_size + uv_size + i * uv_stride,
//            uv_stride);
// }




    // PTS 설정 (프레임 인덱스를 사용하여 PTS 설정)
    // PTS는 각 프레임이 언제 표시되어야하는지를 나타내는 시간정보. 여기서 frame_index는 인코딩중인 프레임 순서 의미
//     frame->pts = frame_index++;                           // 프레임 PTS 설정
    frame->pts = frame_index++ * (codec_ctx->time_base.den / codec_ctx->time_base.num / 8);

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

        if (packet->size > 0 && packet->data) {
            VideoCapture::getInstance().pushImg(packet->data, packet->size);
        }

        //인코딩된 패킷을 출력파일에 기록한 후 패킷을 해제함(unref)
        // av_interleaved_write_fraem 은 인코딩된 패킷을 출력파일에 기록하는 함수
        // 인터리빙 방식 : 오디오 및 비디오 스트림 교차저장하는 방식(영상재생시 동시에 재생되게함)
        // fmt_ctx(포맷컨텍스트) 출력파일과 관련된 정보 포함하고있으며, 파일에 패킷을 기록할 대상이 됨
        // pkt : 인코딩된 비디오또는 오디오 데이터 포함하고있음. 이 데이터를 출력파일에 기록하는것
 //     av_interleaved_write_frame(fmt_ctx, packet);      // 패킷을 파일에 기록
        // 패킷에 할당된 메모리를 해제. 인코딩된 데이터를 파일에 기록한 후 메모리 해제해야함
        av_packet_unref(packet);

    }

}


// 화이트 밸런스 함수
void Camera::applyWhiteBalance(cv::Mat& image) {
    std::vector<cv::Mat> channels;
    cv::split(image, channels); // BGR 채널 분리

    // 각 채널의 평균값 계산
    double bMean = cv::mean(channels[0])[0];
    double gMean = cv::mean(channels[1])[0];
    double rMean = cv::mean(channels[2])[0];

    // 각 채널의 보정 계수 계산
    double k = (bMean + gMean + rMean) / 3.0;
    double bGain = k / bMean;
    double gGain = k / gMean;
    double rGain = k / rMean;

    // 각 채널에 보정 계수 적용
    channels[0] *= bGain; // Blue 채널 보정
    channels[1] *= gGain; // Green 채널 보정
    channels[2] *= rGain; // Red 채널 보정

    // 채널 병합
    cv::merge(channels, image);
}


// RAW 파일을 처리하여 RGB 이미지로 변환하는 함수
void Camera::processRawImageCUDA(void* data, int width, int height) {

    uint16_t* raw = reinterpret_cast<uint16_t*>(data); // RG10 Bayer 데이터

    if (!raw) {
      throw std::runtime_error("Raw data is null");
    }


    // 입력 데이터 크기 (3280x2464)
    int srcWidth = width;
    int srcHeight = height;

    // 출력 데이터 크기 (3264x2464)
    int dstWidth = WIDTH_CROP;
    int dstHeight = srcHeight;

    // GPU 메모리 할당
    uint16_t *d_src, *d_dst;
    size_t srcSize = srcWidth * srcHeight * sizeof(uint16_t);
    size_t dstSize = dstWidth * dstHeight * sizeof(uint16_t);
    cudaMalloc(&d_src, srcSize);
    cudaMalloc(&d_dst, dstSize);

    // 원본 데이터를 GPU 메모리로 복사
    cudaMemcpy(d_src, raw, srcSize, cudaMemcpyHostToDevice);

    // 크롭 및 재구성 작업 수행
    cropAndReorganizeImageCUDA(d_src, d_dst, srcWidth, dstWidth, dstHeight);

    // 결과 데이터를 GpuMat로 래핑
    cv::cuda::GpuMat gpuCroppedRaw(dstHeight, dstWidth, CV_16UC1, d_dst);

//     cv::Mat raw16;
//     gpuCroppedRaw.download(raw16); // gpu → cpu
//     cv::imwrite("raw16.png", raw16);
//     std::cout << "raw16 image size: " << gpuCroppedRaw.cols << "x" << gpuCroppedRaw.rows << std::endl;
//     std::cout << "raw16 type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << gpuCroppedRaw.depth() << std::endl;
//     std::cout << "raw16 number of channels(c1,c3): " << gpuCroppedRaw.channels() << std::endl;


    // 16비트 데이터를 8비트로 변환
    cv::cuda::GpuMat gpu8bitRaw;
    gpuCroppedRaw.convertTo(gpu8bitRaw, CV_8U, 255.0 / 65535.0);

//     cv::Mat raw8;
//     gpu8bitRaw.download(raw8); // gpu → cpu
//     cv::imwrite("gpu8bitRaw.png", raw8);
//     std::cout << "gpu8bitRaw image size: " << gpu8bitRaw.cols << "x" << gpu8bitRaw.rows << std::endl;
//     std::cout << "gpu8bitRaw type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << gpu8bitRaw.depth() << std::endl;
//     std::cout << "gpu8bitRaw number of channels(c1,c3): " << gpu8bitRaw.channels() << std::endl;

    // CUDA를 사용한 디모자이킹 (Debayering)
    cv::cuda::GpuMat gpuBGR;
    //cv::cuda::demosaicing(gpu8bitRaw, gpuRGB, cv::COLOR_BayerRG2BGR); //Demosaicing using bilinear interpolation
    cv::cuda::demosaicing(gpu8bitRaw, gpuBGR, cv::cuda::COLOR_BayerRG2BGR_MHT); //Demosaicing using Malvar-He-Cutler algorithm

//     std::cout << "gpuRGB image size: " << gpuBGR.cols << "x" << gpuBGR.rows << std::endl;
//     std::cout << "gpuBGR step: " << gpuBGR.step << std::endl;
//     std::cout << "gpuRGB type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << gpuBGR.depth() << std::endl;
//     std::cout << "gpuRGB number of channels(c1,c3): " << gpuBGR.channels() << std::endl;

//     cv::Mat Demosaicing;
//     gpuBGR.download(Demosaicing); // GPU → CPU
//     cv::imshow("demosaicing Image", Demosaicing);
//     cv::imwrite("demosaicing.png", Demosaicing);



    // 화이트 밸런스 및 감마 보정 적용 (CUDA 커널 호출)
    float gamma = 0.8f;
    //float rGain = 3.0f, gGain = 0.6f, bGain = 1.1f; // 임의 설정, 필요시 동적으로 조정 가능
    //applyWhiteBalanceAndGammaCUDA(gpuRGB, rGain, gGain, bGain, gamma);
    applyWhiteBalanceAndGammaCUDA(gpuBGR, gamma);

//     cv::Mat whitebalance;
//     gpuBGR.download(whitebalance); // GPU → CPU
//     cv::imshow("white Balance Image", whitebalance);
//     cv::imwrite("white Balance.png", whitebalance);


//     std::cout << "[white] gpuBGR image: " << gpuBGR.cols << "x" << gpuBGR.rows << std::endl;
//     std::cout << "[white] gpuBGR step: " << gpuBGR.step << std::endl;
//     std::cout << "[white] gpuBGR type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << gpuBGR.depth() << std::endl;
//     std::cout << "[white] gpuBGR number of channels(c1,c3): " << gpuBGR.channels() << std::endl << std::endl;


    // CUDA YUV420p 변환 수행
    cv::cuda::GpuMat gpuYUV420p;
    BGRtoYUV420P(gpuBGR, gpuYUV420p);

//     std::cout << "[DEBUG] gpuYUV420p image: " << gpuYUV420p.cols << "x" << gpuYUV420p.rows << std::endl;
//     std::cout << "[DEBUG] gpuYUV420p step: " << gpuYUV420p.step << std::endl;
//     std::cout << "[DEBUG] gpuyuv420p type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << gpuYUV420p.depth() << std::endl;
//     std::cout << "[DEBUG] gpuyuv420p number of channels(c1,c3): " << gpuYUV420p.channels() << std::endl<< std::endl;


    // GPU -> CPU
    cv::Mat cpuYUV420p;
    gpuYUV420p.download(cpuYUV420p);
//     std::cout << "[DEBUG] cpuYUV420p image: " << cpuYUV420p.cols << "x" << cpuYUV420p.rows << std::endl;
//     std::cout << "[DEBUG] cpuYUV420p step: " << cpuYUV420p.step << std::endl;
//     std::cout << "[DEBUG] cpuYUV420p type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << cpuYUV420p.depth() << std::endl;
//     std::cout << "[DEBUG] cpuYUV420p number of channels(c1,c3): " << cpuYUV420p.channels() << std::endl<< std::endl;

    // cuda_outpu.yuv 저장
//     std::ofstream cudaOutFile("cuda_output.yuv", std::ios::binary);
//     if (!cudaOutFile.is_open()) {
//         throw std::runtime_error("Failed to open cuda_output.yuv for writing.");
//     }
//     cudaOutFile.write(reinterpret_cast<const char*>(cpuYUV420p.data), cpuYUV420p.total() * cpuYUV420p.elemSize());
//
//     cudaOutFile.close();
//     std::cout << "CPU YUV420p data saved to cuda_output.yuv" << std::endl;

    // YUV420P에서 BGR로 변환 수행
//     cv::Mat restoredBgrImage;
//     cv::cvtColor(cpuYUV420p, restoredBgrImage, cv::COLOR_YUV2BGR_I420);
//     std::cout << "[DEBUG] restoredBgrImage image: " << restoredBgrImage.cols << "x" << restoredBgrImage.rows << std::endl;
//     std::cout << "[DEBUG] restoredBgrImage step: " << restoredBgrImage.step << std::endl;
//     std::cout << "[DEBUG} restoredBgrImage type 0(cv_8u), 1(cv_8s), 2(cv_16u): " << restoredBgrImage.depth() << std::endl;
//     std::cout << "[DEBUG] restoredBgrImage number of channels(c1,c3): " << restoredBgrImage.channels() << std::endl<< std::endl;

    // GPU에서 CPU로 다운로드 및 시각화
//     cv::imshow("Processed Image", restoredBgrImage);
//     cv::imwrite("Processd.png", restoredBgrImage);

    // FFmpeg로 인코딩
    encodeFrame(cpuYUV420p, cpuYUV420p.total());

    // GPU에서 CPU로 다운로드 및 시각화
//     cv::Mat finalImage;
//     restoredBgrImage.download(finalImage); // GPU → CPU
//     cv::imshow("Processed Image", finalImage);
//     cv::imwrite("Processd.png", finalImage);

    // Step 7: 키 입력으로 종료
//     if (cv::waitKey(0) == 'q') {
//         throw std::runtime_error("Quit");
//     }

    cudaFree(d_src);
    cudaFree(d_dst);

}

//-------------------------------------Using-OpenCV-CUDA-output-end-------------------------------------//



















bool Camera::captureOpencv(Camera& camera) {


    struct v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // 1. 큐에서 버퍼를 가져옴
    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        if (errno == EAGAIN) return false; // 큐에 데이터가 없는 경우
        throw std::runtime_error("Failed to dequeue buffer");
    }


//     // raw 데이터 크기 출력
//     std::cout<<"Size of v4l2_buffer bytesused : " << buf.bytesused << std::endl;
//
//     // raw 데이터 저장
//     saveFrameToFile(buffers[buf.index].start, buf.bytesused);
//
//     // RAW 데이터 분석 및 히스토그램 시각화
//     processAndVisualizeRawData(buffers[buf.index].start, WIDTH, HEIGHT);
//     saveHistogramImage(buffers[buf.index].start, WIDTH, HEIGHT, "histogram.png","histogram.csv");
//
//     try {
//         processRGGBImage(buffers[buf.index].start, WIDTH, HEIGHT, "rggb_histogram.csv","rggb_histogram.png");
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }
//


    // 3. CUDA 또는 일반 OpenCV 처리 함수 호출
    processRawImageCUDA(buffers[buf.index].start, WIDTH, HEIGHT);

    // 4. 버퍼를 다시 큐에 추가
    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        throw std::runtime_error("Failed to queue buffer");
    }

    return true;
}
