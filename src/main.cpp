#include "common.h"
#include "CUDAimageprocessing.h"
#include "GStreamerCapture.h"
#include "FFmpegEncoder.h"
#include "RTSPServer.h"

int main() {
    try {
        const std::string pipeline = "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1920,height=1080,framerate=21/1,format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink";
        GStreamerCapture gstreamerCapture(pipeline);
        FFmpegEncoder ffmpegEncoder("output.h264",WIDTH_RESIZE, HEIGHT_RESIZE, 30.0);

        RTSPServer rtspServer;
        rtspServer.start();

        int frameCount = 0;
        double fps = 30.0;
        double accumulatedFPS = 0.0;
        int fpsMeasurements = 0;

        auto start = std::chrono::high_resolution_clock::now();


        while (true) {

            cv::Mat frame;

            if (!gstreamerCapture.read(frame)) break;

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

            cv::putText(frame, "FPS: " + std::to_string(fps), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 2.0, cv::Scalar(0, 255, 0), 3);


            cv::cuda::GpuMat gpuBGR, gpuYUV420p;
            gpuBGR.upload(frame);
            BGRtoYUV420P(gpuBGR, gpuYUV420p);

            cv::Mat cpuYUV420p;
            gpuYUV420p.download(cpuYUV420p);

            ffmpegEncoder.encode(cpuYUV420p, fps);

            cv::imshow("Frame", frame);
            if (cv::waitKey(1) == 'q') break;

        }

        // OpenCV 창 닫기
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


