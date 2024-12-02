#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>

//카메라 디바이스 경로
#define VIDEODEV "/dev/video0"
#define WIDTH 800
#define HEIGHT 600

// 비디오 데이터 저장할 메모리 버퍼 구조체
struct buffer {
    void *start;
    size_t length;
};

// 버퍼 배열, 버퍼의 수, 카메라 파일 디스크립터
static struct buffer *buffers;
static unsigned int n_buffers;
static int camfd = -1;

// xioctl function to handle ioctl calls with retry on EINTR
// EINTR는 시그널 인터럽트 . ioctl 반복실행하여 EINTR 오류 발생시 다시 시도
static int xioctl(int fd, int request, void *arg) {
    int r;
    do r = ioctl(fd, request, arg); while (r == -1 && errno == EINTR);
    return r;
}

// 카메라 초기화 위한 함수.
//initialize V4L2 and mmap the buffers for video capture
static int init_camera() {
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;

    // Open camera device
    camfd = open(VIDEODEV, O_RDWR | O_NONBLOCK, 0);
    if (camfd == -1) {
        perror("Opening video device");
        return -1;
    }

    // Query device capabilities
    // 비디오 캡쳐 장치인지 확인
    if (xioctl(camfd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("Querying capabilities");
        return -1;
    }

    // Set video format
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (xioctl(camfd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Setting Pixel Format");
        return -1;
    }

      // 4개의 버퍼 요청하여 메모리 매핑방식으로 사용
    // Request buffers for memory mapping
    memset(&req, 0, sizeof(req));
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(camfd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Requesting Buffer");
        return -1;
    }

    // Memory map the buffers
    buffers = calloc(req.count, sizeof(*buffers));
    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = n_buffers;

        if (xioctl(camfd, VIDIOC_QUERYBUF, &buf) == -1) {
            perror("Querying Buffer");
            return -1;
        }

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, camfd, buf.m.offset);
        if (buffers[n_buffers].start == MAP_FAILED) {
            perror("mmap");
            return -1;
        }
    }
      //매핑된 버퍼를 비디오 캡처 대기열에 추가
    // Queue the buffers for capture
    for (unsigned int i = 0; i < n_buffers; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (xioctl(camfd, VIDIOC_QBUF, &buf) == -1) {
            perror("Queue Buffer");
            return -1;
        }
    }
      // 비디오 스트리밍 시작
    // Start video streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(camfd, VIDIOC_STREAMON, &type) == -1) {
        perror("Stream On");
        return -1;
    }

    return 0;
}


void yuyv_to_yuv420p_manual(unsigned char *yuyv, AVFrame *frame, int width, int height) {
    unsigned char *y_plane = frame->data[0];  // Y plane
    unsigned char *u_plane = frame->data[1];  // U plane
    unsigned char *v_plane = frame->data[2];  // V plane

    int y_stride = frame->linesize[0];
    int u_stride = frame->linesize[1];
    int v_stride = frame->linesize[2];

    // YUYV는 2바이트가 1픽셀을 표현: Y0 U Y1 V로 저장
    for (int j = 0; j < height; j += 2) {
        for (int i = 0; i < width; i += 2) {
            // 픽셀 4개씩 처리 (2x2 블록)
            int y_index_00 = (j * width + i) * 2;
            int y_index_01 = (j * width + i + 1) * 2;
            int y_index_10 = ((j + 1) * width + i) * 2;
            int y_index_11 = ((j + 1) * width + i + 1) * 2;

            // 각 채널에 값을 저장 (Y는 그대로, U와 V는 샘플링)
            y_plane[j * y_stride + i] = yuyv[y_index_00];        // Y00
            y_plane[j * y_stride + i + 1] = yuyv[y_index_01];    // Y01
            y_plane[(j + 1) * y_stride + i] = yuyv[y_index_10];  // Y10
            y_plane[(j + 1) * y_stride + i + 1] = yuyv[y_index_11]; // Y11

            // U와 V는 2x2 픽셀당 하나씩 샘플링
            u_plane[(j / 2) * u_stride + (i / 2)] = yuyv[y_index_00 + 1]; // U
            v_plane[(j / 2) * v_stride + (i / 2)] = yuyv[y_index_01 + 1]; // V
        }
    }
}


// YUYV : [Y0 U / Y1 V]
// Y0(한바이트) U(한바이트) 해서 2바이트가 한픽셀
// Y1(한바이트) V(한바이트) 해서 2바이트가 한픽셀
// YUV420p : Y를 모든 픽셀에 저장하고, U와 V는 2*2 픽셀 블록마다 하나씩 공유
// y는 너비*높이 크기로 할당되고 U와 V는 너비/2 * 높이/2 로 할당
// Convert YUYV format to YUV420p
void yuyv_to_yuv420p(unsigned char *yuyv, unsigned char *yuv420p, int width, int height) {
    // Y값을 저장할 위치. 시작주소(yuv420p) 부터 시작
    unsigned char *y_plane = yuv420p;
    // U값을 저장할 위치. Y데이터 담은 후 시작.
    // Y데이터의 크기가 너비*높이만큼 들어가니까 width*height
    unsigned char *u_plane = yuv420p + width * height;
    // V값을 저장할 위치. Y와 U데이터 담은 후 시작.
    // U데이터는 높이/2 * 너비/2 즉 높이*너비/4 니까, V데이터는 높이*너비 + 높이*너비/4 라서 높이*너비*5/4 만큼 더해준거임
    unsigned char *v_plane = yuv420p + width * height * 5 / 4;


      // 픽셀데이터는 2바이트씩 그룹으로 저장됨. 즉 한픽셀당 2바이트이고, YUYV포맷에선 두개픽셀당 4바이트 사용
      // 한 그룹은 [Y0U0] 그다음그룹은 [Y1V0]
      // Y0, Y1은 첫번째와 두번째 픽셀의 밝기값. U0과 V0은 두 픽셀이 공유하는 색차 값
      // 그래서 라인당 바이트 수는 width * 2바이트 (한행의 모든 픽셀 수 * 픽셀당 바이트 수)
      // YUV420p에서는 U와 V값을 2*2 블록마다 하나씩 저장하기때문에 두개의 픽셀 건너뜀
    for (int j = 0; j < height; j += 2) {
        for (int i = 0; i < width; i += 2) {
              // 첫번째 픽셀(i,j) 에서 Y0과 U 값 가져옴
              // 한픽셀당 2바이트이므로 *2를 하고, + 0 한건 첫번째 바이트를 선택한단것
              // 즉 [Y0 U0] 에서 Y0을 선택
            int y0 = yuyv[(j * width + i) * 2 + 0];
            // 이건 U0을 선택한것
            int u = yuyv[(j * width + i) * 2 + 1];
            // i에 +1을 해서 다음 픽셀로 이동
            int y1 = yuyv[(j * width + i + 1) * 2 + 0];
            int v = yuyv[(j * width + i + 1) * 2 + 1];

            // YUV420p는 2*2픽셀의 Y값이 다 필요하므로, 다음행의 Y값도 가져옴
            int y2 = yuyv[((j + 1) * width + i) * 2 + 0];
            int y3 = yuyv[((j + 1) * width + i + 1) * 2 + 0];

                  // Yplane에 먼저 y값 4개 2*2에 맞춰 저장
            y_plane[j * width + i] = y0;
            y_plane[j * width + i + 1] = y1;
            y_plane[(j + 1) * width + i] = y2;
            y_plane[(j + 1) * width + i + 1] = y3;

                  // Uplane과 Vplane에도 알맞게 저장
            u_plane[(j / 2) * (width / 2) + (i / 2)] = u;
            v_plane[(j / 2) * (width / 2) + (i / 2)] = v;
        }
    }
}

// Initialize FFmpeg
// ffmpeg을 초기화하여 h264 인코딩 설정
// avcodec_find_encoder 함수 사용하여 h264코덱 찾음
void initialize_ffmpeg(AVCodecContext **codec_ctx, AVFormatContext **fmt_ctx, const char *filename) {
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    // 코덱컨텍스트 할당
    *codec_ctx = avcodec_alloc_context3(codec);
    if (!*codec_ctx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }

   // 초당 처리되는 비트 양 설정(400,000 bits per second 즉 400kbps)
    (*codec_ctx)->bit_rate = 400000;
    // 인코딩할 비디오의 해상도 설정(가로 세로)
    (*codec_ctx)->width = WIDTH;
    (*codec_ctx)->height = HEIGHT;
    // 비디오 프레임의 시간 단위 설정
    //AVRational 구조체는 분수형태 즉 1/25초의 시간 걸린다는의미
    // 프레임레이트가 25fps라는 뜻 즉 1초당 25개 프레임 재생
    (*codec_ctx)->time_base = (AVRational){1, 25};
    // I프레임과 P/B프레임간의 간격 설정 (GOP:Group of Pictures)
    // I프레임은 전체화면저장하는 완전한 프레임이고, P/B프레임은 이전 또는 다음프레임에 의존하는 차이프레임
    // gop_size = 10은 매 10프레임마다 I프레임을 삽입하는것
    (*codec_ctx)->gop_size = 10;
    // B프레임은 I 또는 P프레임 사이에 위치하며, 이전 및 이후 프레임 참조하여 압축 극대화하는 프레임
    // 여기서 최대 1개의 B프레임을 사용 (적게사용시 인코딩/디코딩 빨라지지만 압축 효율 낮아짐)
    (*codec_ctx)->max_b_frames = 1;
    // 비디오의 픽셀 포맷 YUV420p 포맷 사용
    (*codec_ctx)->pix_fmt = AV_PIX_FMT_YUV420P;

   // 코덱 열고 초기화. 실패시 에러 출력
    if (avcodec_open2(*codec_ctx, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

   // ffmpeg에서 사용할 포맷 컨텍스트 할당
    *fmt_ctx = avformat_alloc_context();
    if (!*fmt_ctx) {
        fprintf(stderr, "Could not allocate format context\n");
        exit(1);
    }

   // 출력파일 포맷 결정 (파일 이름에 따라 적절한 포맷 추측)
    const AVOutputFormat *output_fmt = av_guess_format(NULL, filename, NULL);
    (*fmt_ctx)->oformat = output_fmt;

    //avio_open으로 파일 열기. 쓰기모드로 염
    //파일입출력을위한 I/O핸들러를 여는 함수. 주어진 파일을 쓰기모드로 열고, FFmpeg이 해당파일에 데이터 쓸 수 있게함
    // AVFormatContext구조체의 pb필드는 입출력 핸들임
    if (avio_open(&(*fmt_ctx)->pb, filename, AVIO_FLAG_WRITE) < 0) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(1);
    }

   // ffmpeg라이브러리로 새 비디오 스트림 만들고 스트림 다양한 속성 설정 후 avformat_write_header로 파일헤더작성
    // 새 비디오 스트림 생성(avformat_new_stream함수는 AVFormatContext(fmt_ctx)에 새로운 AVStream (오디오 등등..)추가)
    // 두번째 인자는 사용할 코덱 명시 (NULL사용하여 기본코덱 사용하겠다는 의미)
    AVStream *stream = avformat_new_stream(*fmt_ctx, NULL);
      // 스트림의 시간기준 설정. 프레임속도 25fps
    stream->time_base = (AVRational){1, 25};
    // 스트림에 사용할 비디오코덱 설정. h264로 인코딩될것
    stream->codecpar->codec_id = AV_CODEC_ID_H264;
    // 스트림의 미디어타입 설정. 비디오스트림과 오디오스트림 구분시 사용
    stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    stream->codecpar->width = WIDTH;
    stream->codecpar->height = HEIGHT;
    // 비디오의 픽셀 포맷 설정
    stream->codecpar->format = AV_PIX_FMT_YUV420P;

      // 비디오 파일의 헤더 작성
    avformat_write_header(*fmt_ctx, NULL);


    // 코덱 컨텍스트는 비디오나 오디오 데이터가 어떻게 압축되거나 풀리는지에 대한 정보 저장함.
    // H.264, AAC 등으로 압축할때 코덱 컨텍스트 설정하여 데이터 인코딩

    // 포맷 컨텍스트는 비디오나 오디오 담고있는 컨테이너 포맷 처리시 사용. 컨테이너포맷은 비디오와 오디오 데이터 뿐만 아니라 자막,메타데이터 등등 다양한정보 함께 담고있음
    // MP4, MKV 등등의 정보와 데이터 관리
}

// Encode a frame using FFmpeg
// 하나의 프레임을 인코딩
void encode_frame(AVCodecContext *codec_ctx, AVFormatContext *fmt_ctx, AVFrame *frame, AVPacket *pkt) {
   // avcodec_send_frame 을 사용해서 프레임을 인코더로 보냄
    int ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending frame for encoding\n");
        exit(1);
    }

    while (ret >= 0) {
       // avcodec_receive_packet() 으로 인코딩된 데이터를 패킷으로 받음
        ret = avcodec_receive_packet(codec_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) return;
        if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
        //인코딩된 패킷을 출력파일에 기록한 후 패킷을 해제함(unref)
        // av_interleaved_write_fraem 은 인코딩된 패킷을 출력파일에 기록하는 함수
        // 인터리빙 방식 : 오디오 및 비디오 스트림 교차저장하는 방식(영상재생시 동시에 재생되게함)
        // fmt_ctx(포맷컨텍스트) 출력파일과 관련된 정보 포함하고있으며, 파일에 패킷을 기록할 대상이 됨
        // pkt : 인코딩된 비디오또는 오디오 데이터 포함하고있음. 이 데이터를 출력파일에 기록하는것
        av_interleaved_write_frame(fmt_ctx, pkt);
        // 패킷에 할당된 메모리를 해제. 인코딩된 데이터를 파일에 기록한 후 메모리 해제해야함
        av_packet_unref(pkt);
    }
}

// 카메라로부터 프레임을 읽어 인코딩
// Main loop to read frames and encode
void read_frame_and_encode(int fd, AVCodecContext *codec_ctx, AVFormatContext *fmt_ctx, AVFrame *frame, AVPacket *pkt, int frame_index) {
    struct v4l2_buffer buf;
    fd_set fds;
    struct timeval tv;
    int r;

    // Set up the file descriptor set.
    // select 함수가 기다릴 파일 디스크립터를 파일디스크립터집합(fds)에 추가할수있게 집합을 초기화
    FD_ZERO(&fds);
    // fds(파일디스크립터집합)에 파일디스크립터(fd) 추가. 이후 select에 의해 감시가능
    FD_SET(fd, &fds);

    // Set up the timeout.
    tv.tv_sec = 2;  // 2 seconds timeout 2초 지나도 감시중인 파일디스크립터에 이벤트 발생 x라면 타임아웃발생
    tv.tv_usec = 0;

    // Wait for the camera to be ready for reading
    r = select(fd + 1, &fds, NULL, NULL, &tv);
    if (r == -1) {
        perror("select");
        return;
    } else if (r == 0) {
        fprintf(stderr, "select timeout\n");
        return;
    }

    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    for (int retry = 0; retry < 5; ++retry) {
        if (xioctl(fd, VIDIOC_DQBUF, &buf) != -1) {
            break;  // 성공적으로 읽었을 때 반복 종료
        }
        usleep(10000);  // 재시도 전 대기
        printf("retry %d \n", retry);
    }

    unsigned char *yuv420p_data = (unsigned char *)malloc(WIDTH * HEIGHT * 3 / 2);
    //yuyv_to_yuv420p((unsigned char *)buffers[buf.index].start, yuv420p_data, WIDTH, HEIGHT);
    yuyv_to_yuv420p_manual((unsigned char *)buffers[buf.index].start, frame, WIDTH, HEIGHT);
   //av_frame_make_writable로 프레임 쓸 수 있게 함
    if (av_frame_make_writable(frame) < 0) {
        fprintf(stderr, "Frame not writable\n");
        exit(1);
    }

   // 변환된 데이터를 ffmpeg프레임의 각 채널에 복사
    //memcpy(frame->data[0], yuv420p_data, WIDTH * HEIGHT);
    //memcpy(frame->data[2], yuv420p_data + WIDTH * HEIGHT, WIDTH * HEIGHT / 4);
    //memcpy(frame->data[1], yuv420p_data + WIDTH * HEIGHT * 5 / 4, WIDTH * HEIGHT / 4);

    // PTS 설정 (프레임 인덱스를 사용하여 PTS 설정)
    // PTS는 각 프레임이 언제 표시되어야하는지를 나타내는 시간정보. 여기서 frame_index는 인코딩중인 프레임 순서 의미
    frame->pts = frame_index;
    encode_frame(codec_ctx, fmt_ctx, frame, pkt);

    if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        perror("VIDIOC_QBUF");
        return;
    }

   //동적으로 할당한 YUV420p데이터 해제
    free(yuv420p_data);
}

int main() {
    // Initialize camera
    if (init_camera() != 0) {
        fprintf(stderr, "Failed to initialize camera\n");
        return -1;
    }

    // Initialize FFmpeg
    AVCodecContext *codec_ctx;
    AVFormatContext *fmt_ctx;
    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Could not allocate AVPacket\n");
        return -1;
    }

    AVFrame *frame;
    const char *output_filename = "output.h264";
    initialize_ffmpeg(&codec_ctx, &fmt_ctx, output_filename);

    frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = codec_ctx->width;
    frame->height = codec_ctx->height;

    if (av_frame_get_buffer(frame, 32) < 0) {
        fprintf(stderr, "Could not allocate the video frame data\n");
        exit(1);  // 버퍼 할당 실패 시 프로그램 종료
    }

    // Main loop for capturing frames and encoding
    for (int frame_index = 0; frame_index < 500; ++frame_index) {
        read_frame_and_encode(camfd, codec_ctx, fmt_ctx, frame, pkt, frame_index);
    }

    // Finalize FFmpeg
    av_write_trailer(fmt_ctx);
    avcodec_close(codec_ctx);
    avio_close(fmt_ctx->pb);
    avformat_free_context(fmt_ctx);
    av_packet_free(&pkt);

    // Stop camera capture and clean up
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(camfd, VIDIOC_STREAMOFF, &type);
    for (unsigned int i = 0; i < n_buffers; ++i) {
        munmap(buffers[i].start, buffers[i].length);
    }
    close(camfd);

    return 0;
}

