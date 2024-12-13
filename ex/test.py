import cv2
import time

# GStreamer를 이용해서 카메라로부터 영상을 가져오고, flip-method 수정
cap = cv2.VideoCapture(
    "nvarguscamerasrc ! nvvidconv flip-method=2 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink",
    cv2.CAP_GSTREAMER
)

if cap.isOpened():
    window_handle = cv2.namedWindow("VideoFrame", cv2.WINDOW_AUTOSIZE)
    prev_time = time.time()  # 초기 시간 설정

    while cv2.getWindowProperty("VideoFrame", 0) >= 0:
        ret_val, frame = cap.read()
        if not ret_val:
            print("Frame read error!")
            break

        # 현재 시간과 이전 시간 차이를 계산하여 FPS를 측정
        curr_time = time.time()
        elapsed_time = curr_time - prev_time
        prev_time = curr_time
        fps = 1.0 / elapsed_time

        # FPS 값을 화면에 표시
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) > 0:  # 키 입력 감지
            break

cap.release()
cv2.destroyAllWindows()

