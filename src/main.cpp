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

        // ===== CUDA 변환 =====
        cv::cuda::GpuMat gpuRGB, gpuY, gpuU, gpuV;
        gpuRGB.upload(inputRGB);

        // CUDA YUV420p 변환 수행
        rgbToYUV420pCUDA(gpuRGB, gpuY, gpuU, gpuV);

        // GPU 결과를 CPU로 다운로드
        cv::Mat cudaY, cudaU, cudaV;
        gpuY.download(cudaY);
        gpuU.download(cudaU);
        gpuV.download(cudaV);

        // CUDA YUV420p 데이터를 병합
        cv::Mat cudaYUV420p(cudaY.rows * 3 / 2, cudaY.cols, CV_8UC1);
        memcpy(cudaYUV420p.data, cudaY.data, cudaY.rows * cudaY.cols);
        memcpy(cudaYUV420p.data + cudaY.rows * cudaY.cols, cudaU.data, cudaU.rows * cudaU.cols);
        memcpy(cudaYUV420p.data + cudaY.rows * cudaY.cols + cudaU.rows * cudaU.cols, cudaV.data, cudaV.rows * cudaV.cols);

        // CUDA 변환 결과 파일 저장
        std::ofstream cudaOutFile("cuda_output.yuv", std::ios::binary);
        if (!cudaOutFile.is_open()) {
            throw std::runtime_error("Failed to open cuda_output.yuv for writing.");
        }
        cudaOutFile.write(reinterpret_cast<const char*>(cudaYUV420p.data), cudaYUV420p.total());
        cudaOutFile.close();
        std::cout << "CUDA YUV420p data saved to cuda_output.yuv" << std::endl;

        // ===== G-API 변환 =====
        cv::GMat in;
        cv::GMat out = cv::gapi::BGR2I420(in);
        auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(out));

        // G-API 실행
        cv::Mat gapiYUV420p;
        pipeline.apply(cv::gin(inputRGB), cv::gout(gapiYUV420p));

        // G-API 변환 결과 파일 저장
        std::ofstream gapiOutFile("gapi_output.yuv", std::ios::binary);
        if (!gapiOutFile.is_open()) {
            throw std::runtime_error("Failed to open gapi_output.yuv for writing.");
        }
        gapiOutFile.write(reinterpret_cast<const char*>(gapiYUV420p.data), gapiYUV420p.total());
        gapiOutFile.close();
        std::cout << "G-API YUV420p data saved to gapi_output.yuv" << std::endl;

        // ===== CUDA와 G-API 결과 비교 =====
        if (cudaYUV420p.total() != gapiYUV420p.total()) {
            throw std::runtime_error("CUDA and G-API outputs have different sizes.");
        }

        if (memcmp(cudaYUV420p.data, gapiYUV420p.data, cudaYUV420p.total()) == 0) {
            std::cout << "CUDA and G-API outputs are identical!" << std::endl;
        } else {
            std::cout << "CUDA and G-API outputs differ!" << std::endl;

            int numDifferences = 10;
            // 디버깅용: 차이가 나는 첫 10개 값 출력
            for (size_t i = 0; i < cudaYUV420p.total(); ++i) {
                if (cudaYUV420p.data[i] != gapiYUV420p.data[i]) {
                    std::cout << "Difference at index " << i
                              << ": CUDA=" << static_cast<int>(cudaYUV420p.data[i])
                              << ", G-API=" << static_cast<int>(gapiYUV420p.data[i]) << std::endl;

                    if (--numDifferences == 0) break; // 최대 10개만 출력
                }
            }
        }

        // ===== YUV420P -> BGR 변환 및 출력 =====
        // CUDA 결과
        cv::Mat cudaBGR;
        cv::cvtColor(cudaYUV420p, cudaBGR, cv::COLOR_YUV2BGR_I420);
        cv::imshow("CUDA Restored BGR", cudaBGR);
        cv::imwrite("CUDA_RestoredBGR.png", cudaBGR);

        // G-API 결과
        cv::Mat gapiBGR;
        cv::cvtColor(gapiYUV420p, gapiBGR, cv::COLOR_YUV2BGR_I420);
        cv::imshow("G-API Restored BGR", gapiBGR);
        cv::imwrite("GAPI_RestoredBGR.png", gapiBGR);

        std::cout << "Restored BGR images saved to CUDA_RestoredBGR.png and GAPI_RestoredBGR.png" << std::endl;

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
