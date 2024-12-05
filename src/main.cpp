#include "common.h"
#include "FrameBuffer.h"
#include "Camera.h"
#include "ClientSession.h"
#include "TCPHandler.h"

int main() {
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


    //**** TODO : example) rtsp로 전송할 이미지를 VideoCapture Queue로 던지기 ***** */
   // std::pair<const unsigned char*, const unsigned int> cur_frame = h264_file->get_next_frame();
   // unsigned char* ptr_cur_frame = cur_frame.first;
   // unsigned int cur_frame_size = cur_frame.second;
   // VideoCapture::getInstance().pushImg((unsigned char *)ptr_cur_frame, cur_frame_size);
    //********************************************************* */

    try
    {
        Camera camera;
        camera.initFFmpeg("output.h264"); // FFmpeg 초기화: H.264 파일 준비
//
        // camera.checkFormat();
        for (int i = 0; i < 500; ++i) {
            for (;;) {  /* 내부 무한 루프: 성공적으로 프레임을 읽을 때까지 반복 */
                fd_set fds;  /* 파일 디스크립터 셋을 선언: select()로 이벤트를 감시할 파일 디스크립터 */
                struct timeval tv;  /* 타임아웃을 설정하기 위한 구조체 */
//
                FD_ZERO(&fds);  /* fd_set을 초기화: 모든 비트가 0으로 설정됨 */
                FD_SET(camera.get_fd(), &fds);  /* 비디오 장치 파일 디스크립터(fd)를 fd_set에 추가 */
                /* 타임아웃 설정: 최대 2초 동안 대기 */
                tv.tv_sec = 2;  /* 초 단위 타임아웃 설정 (2초 대기) */
                tv.tv_usec = 0;  /* 마이크로초 단위 타임아웃 설정 (0마이크로초) */
//
                /* select() 호출: 파일 디스크립터에서 이벤트가 발생할 때까지 대기 */
                int r = select(camera.get_fd() + 1, &fds, NULL, NULL, &tv);
                /* select()는 파일 디스크립터에서 읽기 가능 상태가 될 때까지 대기 (또는 타임아웃) */
                if (-1 == r) {  /* select() 호출이 실패한 경우 */
                    throw std::runtime_error("select");
                } else if (0 == r) {  /* select() 호출이 타임아웃으로 인해 반환된 경우 */
                    throw std::runtime_error("select timeout");
                }
//
                /* 프레임을 읽음: read_frame() 함수를 호출해 프레임을 읽음 */
                //if (camera.captureFrame(framebuffer)) break;  /* 프레임이 성공적으로 읽히면 무한 루프를 종료하고 다음 프레임 처리로 이동 */
//
//              usleep(20000);
//
                if (camera.captureOpencv(camera)) break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
//
    return EXIT_SUCCESS;
}
