#include "common.h"
#include "FrameBuffer.h"
#include "Camera.h"
#include "CUDAimageprocessing.h"

int main() {
    try {


        const int width = 3264;  // 이미지의 가로 크기
        const int height = 2464; // 이미지의 세로 크기

        // 1. 테스트용 RGB 이미지 로드
        cv::Mat inputRGB = cv::imread("WhiteBalanceAndGamma.png");
        if (inputRGB.empty()) {
            throw std::runtime_error("Failed to load test image.");
        }

//         // 입력 이미지를 GPU로 업로드
//         cv::cuda::GpuMat gpuRGB;
//         gpuRGB.upload(inputRGB);
//
//         // CUDA YUV420p 변환 결과 저장할 변수
//         cv::cuda::GpuMat gpuY, gpuU, gpuV;
//
//         // GPU에서 RGB -> YUV420p 변환 수행
//         rgbToYUV420pCUDA(gpuRGB, gpuY, gpuU, gpuV);
//
//         // GPU 결과를 CPU로 다운로드
//         cv::Mat yPlane, uPlane, vPlane;
//         gpuY.download(yPlane);
//         gpuU.download(uPlane);
//         gpuV.download(vPlane);
//
//         // YUV420p 데이터를 하나의 연속된 Mat로 병합
//         cv::Mat yuv420p(yPlane.rows * 3 / 2, yPlane.cols, CV_8UC1);
//         memcpy(yuv420p.data, yPlane.data, yPlane.rows * yPlane.cols);
//         memcpy(yuv420p.data + yPlane.rows * yPlane.cols, uPlane.data, uPlane.rows * uPlane.cols);
//         memcpy(yuv420p.data + yPlane.rows * yPlane.cols + uPlane.rows * uPlane.cols, vPlane.data, vPlane.rows * uPlane.cols);
//

            // OpenCV G-API로 YUV420P 변환 그래프 설정
        cv::GMat in;
        cv::GMat out = cv::gapi::BGR2I420(in);

        auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out));

        // G-API 실행
        cv::Mat yuv420p;
        pipeline.apply(cv::gin(inputRGB), cv::gout(yuv420p));


        // 병합된 YUV420p 데이터를 파일로 저장
        std::ofstream outFile("output.yuv", std::ios::binary);
        if (!outFile.is_open()) {
            throw std::runtime_error("Failed to open output file.");
        }
        outFile.write(reinterpret_cast<const char*>(yuv420p.data), yuv420p.total());
        outFile.close();
        std::cout << "YUV420p data saved to output.yuv" << std::endl;

        // 2. 저장한 output.yuv 파일을 다시 로드
        std::ifstream inFile("output.yuv", std::ios::binary);
        if (!inFile.is_open()) {
            throw std::runtime_error("Failed to open YUV file.");
        }

        // YUV420P 데이터를 메모리에 로드
        std::vector<uint8_t> yuvData(width * height * 3 / 2);
        inFile.read(reinterpret_cast<char*>(yuvData.data()), yuvData.size());
        inFile.close();

        // YUV 데이터를 OpenCV Mat으로 로드
        cv::Mat yuv420pMat(height * 3 / 2, width, CV_8UC1, yuvData.data());

        // YUV420P -> BGR 변환
        cv::Mat bgrImage;
        cv::cvtColor(yuv420pMat, bgrImage, cv::COLOR_YUV2BGR_I420);

        // 변환된 BGR 이미지 출력
        cv::imshow("BGR Image from YUV", bgrImage);

        // 변환된 이미지를 파일로 저장
        cv::imwrite("RestoredBGR.png", bgrImage);
        std::cout << "Restored BGR image saved to RestoredBGR.png" << std::endl;

        // 사용자 입력 대기
        cv::waitKey(0);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// int main() {
// //
//   try {
//         Camera camera;
//         camera.initFFmpeg("output.h264"); // FFmpeg 초기화: H.264 파일 준비
//
//         // camera.checkFormat();
//         for (int i = 0; i < 100; ++i) {
//             for (;;) {  /* 내부 무한 루프: 성공적으로 프레임을 읽을 때까지 반복 */
//                 fd_set fds;  /* 파일 디스크립터 셋을 선언: select()로 이벤트를 감시할 파일 디스크립터 */
//                 struct timeval tv;  /* 타임아웃을 설정하기 위한 구조체 */
// //
//                 FD_ZERO(&fds);  /* fd_set을 초기화: 모든 비트가 0으로 설정됨 */
//                 FD_SET(camera.get_fd(), &fds);  /* 비디오 장치 파일 디스크립터(fd)를 fd_set에 추가 */
//                 /* 타임아웃 설정: 최대 2초 동안 대기 */
//                 tv.tv_sec = 2;  /* 초 단위 타임아웃 설정 (2초 대기) */
//                 tv.tv_usec = 0;  /* 마이크로초 단위 타임아웃 설정 (0마이크로초) */
// //
//                 /* select() 호출: 파일 디스크립터에서 이벤트가 발생할 때까지 대기 */
//                 int r = select(camera.get_fd() + 1, &fds, NULL, NULL, &tv);
//                 /* select()는 파일 디스크립터에서 읽기 가능 상태가 될 때까지 대기 (또는 타임아웃) */
//                 if (-1 == r) {  /* select() 호출이 실패한 경우 */
//                     throw std::runtime_error("select");
//                 } else if (0 == r) {  /* select() 호출이 타임아웃으로 인해 반환된 경우 */
//                     throw std::runtime_error("select timeout");
//                 }
// //
//                 /* 프레임을 읽음: read_frame() 함수를 호출해 프레임을 읽음 */
//                 //if (camera.captureFrame(framebuffer)) break;  /* 프레임이 성공적으로 읽히면 무한 루프를 종료하고 다음 프레임 처리로 이동 */
//
//
//                 if (camera.captureOpencv(camera)) break;
//             }
//         }
//
//         // FFmpeg 종료 작업
//         av_write_trailer(camera.fmt_ctx);  // 트레일러 작성
//         avcodec_close(camera.codec_ctx);   // 코덱 닫기
//         avio_close(camera.fmt_ctx->pb);    // 파일 닫기
//         avformat_free_context(camera.fmt_ctx); // 포맷 컨텍스트 해제
//
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return EXIT_FAILURE;
//     }
// //
//     return EXIT_SUCCESS;
// }
