#include "CUDAimageprocessing.h"


//----------------------------------------------------- CUDA 커널 정의 시작  ------------------------------------------------
__global__ void BGRtoYUV420PKernel(const uchar* bgr, uchar* yuv, uint64_t width,
    uint64_t height, uint64_t bgr_step, int64_t yuv_step)
{
    uint64_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x == 0 && y == 0) { // 한 번만 출력하기 위해 조건 추가
//         printf("Size of int: %u\n", (unsigned int)sizeof(int));
//         printf("Size of uint64_t: %llu\n", (unsigned long long)sizeof(uint64_t));
//         printf("Size of size_t: %llu\n", (unsigned long long)sizeof(size_t));
//         printf("Size of uchar: %u\n", (unsigned int)sizeof(uchar));
//         printf("Size of float: %u\n", (unsigned int)sizeof(float));
//     }


    if (x < width && y < height)
    {
        uint64_t idx = y*bgr_step + (3 * x);
        uchar B = bgr[idx];
        uchar G = bgr[idx + 1];
        uchar R = bgr[idx + 2];

        // YUV 변환 공식
        float Y = 0.257f * R + 0.504f * G + 0.098f * B + 16;
        float U = -0.148f * R - 0.291f * G + 0.499f * B + 128;
        float V = 0.439f * R - 0.368f * G - 0.071f * B + 128;

        uint64_t y_idx = y * yuv_step + x;
        yuv[y_idx] = static_cast<uchar>(Y);  // Y 채널 저장

        // U, V 채널은 2x2 블록마다 하나의 값을 가짐 (서브샘플링)
        if ((x % 2 == 0) && (y % 2 == 0) && (x < width) && (y < height))
        {
            uint64_t uv_x = x / 2;
            uint64_t uv_y = y / 2;

            // U와 V 채널의 시작 위치 계산
            uint64_t u_offset = yuv_step * height;              // U 채널의 시작 위치 (Y 채널 뒤)
            uint64_t v_offset = u_offset + (yuv_step / 2) * (height / 2);  // V 채널의 시작 위치 (U 채널 뒤)

            // 패딩을 고려한 U와 V 인덱스 계산 및 값 저장
            uint64_t uv_idx = (uv_y/2)*yuv_step + (uv_y%2)*(width/2)  + uv_x;

            // U와 V 값 저장
            yuv[u_offset + uv_idx] = static_cast<uchar>(U);  // U 채널 저장
            yuv[v_offset + uv_idx] = static_cast<uchar>(V);  // V 채널 저장
        }
    }
}




__device__ float blockReduceSum(float val) {
    extern __shared__ float shared[]; // 공유 메모리 크기를 동적으로 설정
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 2D 블록에서 1D 인덱스 계산

    shared[tid] = val; // 각 스레드의 값을 공유 메모리에 저장
    __syncthreads();

    // 병렬 합산
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    // 최종 합계는 shared[0]에 저장
    return (tid == 0) ? shared[0] : 0;
}



// 이미지의 각 채널 평균 계산 커널
__global__ void computeChannelMean(const unsigned char* image, int width, int height,
    uint32_t step, float* mean) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 패딩을 고려한 인덱싱
    int idx = y * step + x * 3;  // 3바이트(RGB)씩 이동

    float bSum = 0.0f, gSum = 0.0f, rSum = 0.0f;

    bSum = image[idx];       // Blue 채널 값
    gSum = image[idx + 1];   // Green 채널 값
    rSum = image[idx + 2];   // Red 채널 값


    // 블록 내 합산
    float bBlockSum = blockReduceSum(bSum);
    float gBlockSum = blockReduceSum(gSum);
    float rBlockSum = blockReduceSum(rSum);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(&mean[0], bBlockSum); // Blue 평균
        atomicAdd(&mean[1], gBlockSum); // Green 평균
        atomicAdd(&mean[2], rBlockSum); // Red 평균
    }

}



// 화이트 밸런스 및 감마 보정 커널
__global__ void whiteBalanceAndGammaKernel(unsigned char* image, int width, int
    height, uint32_t step, float rGain, float gGain, float bGain, float gamma) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * step + x*3;


    // 각 채널 값을 가져옵니다
    float b = image[idx] * bGain;       // Blue 채널
    float g = image[idx + 1] * gGain;   // Green 채널
    float r = image[idx + 2] * rGain;   // Red 채널

    // 감마 보정
    b = powf(b / 255.0f, gamma) * 255.0f;
    g = powf(g / 255.0f, gamma) * 255.0f;
    r = powf(r / 255.0f, gamma) * 255.0f;

    // 값 클램핑
    image[idx] = min(max((int)b, 0), 255);
    image[idx + 1] = min(max((int)g, 0), 255);
    image[idx + 2] = min(max((int)r, 0), 255);

}





__global__ void cropAndReorganizeImageKernel(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 현재 행
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 현재 열

    if (row >= height || col >= dstWidth) return;

    if (row % 2 == 0) {
        // 짝수 행: 3264개의 RG 픽셀만 복사
        if (col < 3264) {
            int srcIdx = row * srcWidth + col;
            int dstIdx = row * dstWidth + col;
            dst[dstIdx] = src[srcIdx];
        }
    } else {
        // 홀수 행: 16개의 의미 있는 GB 픽셀 + 3248개의 GB 픽셀 복사
        int srcIdx;
        if (col < 16) {
            srcIdx = row * srcWidth + 3264 + col; // 의미 있는 GB
        } else if (col < 3264) {
            srcIdx = row * srcWidth + (col - 16); // GB
        } else {
            return; // 범위를 벗어난 경우
        }
        int dstIdx = row * dstWidth + col;
        dst[dstIdx] = src[srcIdx];
    }
}

//----------------------------------------------------- CUDA 커널 정의 끝  ------------------------------------------------















//----------------------------------------------------- CUDA 커널 Call 함수  정의 시작  -----------------------------------


void BGRtoYUV420P(const cv::cuda::GpuMat& bgrImage, cv::cuda::GpuMat& yuvImage)
{
    // 이미지 크기 가져오기
    uint64_t width = bgrImage.cols;
    uint64_t height = bgrImage.rows;
    // 할당된 크기 확인
//     printf("[DEBUG] bgrImage.cols: %d\n", bgrImage.cols);
//     printf("[DEBUG] bgrImage.rows: %d\n", bgrImage.rows);
//     printf("[DEBUG] bgrImage.step: %zu\n",bgrImage.step);


    // 이미지 리사이즈 (해상도가 너무 클 경우 줄이기)
    cv::cuda::GpuMat resizedBgrImage;
    if (width > WIDTH_RESIZE || height > HEIGHT_RESIZE) {  // 해상도가 Full HD 이상인 경우
        cv::cuda::resize(bgrImage, resizedBgrImage, cv::Size(WIDTH_RESIZE, HEIGHT_RESIZE));
        //printf("[DEBUG] 리사이즈된 이미지 크기: %d x %d\n", resizedBgrImage.cols, resizedBgrImage.rows);
        width = resizedBgrImage.cols;
        height = resizedBgrImage.rows;
    } else {
        resizedBgrImage = bgrImage;
    }
    // 할당된 크기 확인
//     printf("[DEBUG] resizedBgrImage.cols: %d\n", resizedBgrImage.cols);
//     printf("[DEBUG] resizedBgrImage.rows: %d\n", resizedBgrImage.rows);
//     printf("[DEBUG] resizedBgrImage.step: %zu\n", resizedBgrImage.step);



    // 출력 이미지는 Y, U, V를 각각 하나의 채널로 갖는 구조 (YUV420P 형식)
    yuvImage.create(height * 3 / 2, width, CV_8UC1);
    // 메모리 할당 확인
    if (yuvImage.empty()) {
        printf("YUV 이미지 메모리 할당에 실패했습니다.\n");
        return;
    }

    // 할당된 크기 확인
//     printf("[DEBUG] yuvImage.cols: %d\n", yuvImage.cols);
//     printf("[DEBUG] yuvImage.rows: %d\n", yuvImage.rows);
//     printf("[DEBUG] yuvImage.step: %zu\n", yuvImage.step);


    // CUDA 커널 설정
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // CUDA 커널 호출
    BGRtoYUV420PKernel<<<numBlocks,
      threadsPerBlock>>>(resizedBgrImage.ptr<uchar>(), yuvImage.ptr<uchar>(),width, height, resizedBgrImage.step, yuvImage.step);

    // CUDA 동기화 - 커널 실행 완료 대기
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 할당된 크기 확인
//     printf("[DEBUG] yuvImage.cols: %d\n", yuvImage.cols);
//     printf("[DEBUG] yuvImage.rows: %d\n", yuvImage.rows);
//     printf("[DEBUG] yuvImage.step: %zu\n", yuvImage.step);


}




// CUDA 기반 화이트 밸런스 및 감마 보정 적용 함수
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float gamma) {
    dim3 block(16, 16);
    dim3 grid((gpuImage.cols + block.x - 1) / block.x, (gpuImage.rows + block.y - 1) / block.y);

//     // 할당된 크기 확인
//     printf("[DEBUG] gpuImage.cols: %d\n", gpuImage.cols);
//     printf("[DEBUG] gpuImage.rows: %d\n", gpuImage.rows);
//     printf("[DEBUG] gpuImage.step: %zu\n", gpuImage.step);

    unsigned char* d_image = gpuImage.data;

    // 채널별 평균값 계산
    float* d_mean;
    cudaMalloc(&d_mean, 3 * sizeof(float));
    cudaMemset(d_mean, 0, 3 * sizeof(float));

    computeChannelMean<<<grid, block>>>(d_image, gpuImage.cols, gpuImage.rows, (uint32_t)gpuImage.step, d_mean);
    cudaDeviceSynchronize();

//     // 할당된 크기 확인
//     printf("[DEBUG] gpuImage.cols: %d\n", gpuImage.cols);
//     printf("[DEBUG] gpuImage.rows: %d\n", gpuImage.rows);
//     printf("[DEBUG] gpuImage.step: %zu\n", gpuImage.step);


    float h_mean[3];
    cudaMemcpy(h_mean, d_mean, 3 * sizeof(float), cudaMemcpyDeviceToHost);

    float bMean = h_mean[0] / (gpuImage.cols * gpuImage.rows);
    float gMean = h_mean[1] / (gpuImage.cols * gpuImage.rows);
    float rMean = h_mean[2] / (gpuImage.cols * gpuImage.rows);

    // 보정 계수 계산
    float k = (bMean + gMean + rMean) / 3.0f;
    float bGain = k / bMean;
    float gGain = k / gMean;
    float rGain = k / rMean;

    // 화이트 밸런스 및 감마 보정 커널 호출
    whiteBalanceAndGammaKernel<<<grid, block>>>(d_image, gpuImage.cols, gpuImage.rows, (uint32_t)gpuImage.step, bGain, gGain, rGain, gamma);
    cudaDeviceSynchronize();

//     // 할당된 크기 확인
//     printf("[DEBUG] gpuImage.cols: %d\n", gpuImage.cols);
//     printf("[DEBUG] gpuImage.rows: %d\n", gpuImage.rows);
//     printf("[DEBUG] gpuImage.step: %zu\n", gpuImage.step);


    cudaFree(d_mean);
}




// CUDA 기반 크롭 및 재구성 함수
void cropAndReorganizeImageCUDA(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height) {
    dim3 block(256); // 스레드 블록 크기
    dim3 grid((dstWidth + block.x - 1) / block.x, height); // 각 행을 처리하는 그리드 크기

    // CUDA 커널 호출
    cropAndReorganizeImageKernel<<<grid, block>>>(src, dst, srcWidth, dstWidth, height);
    cudaDeviceSynchronize();
}


//----------------------------------------------------- CUDA 커널 Call 함수 정의 끝  -----------------------------------
