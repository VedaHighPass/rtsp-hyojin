#ifndef CUDA_IMAGE_PROCESSING_H
#define CUDA_IMAGE_PROCESSING_H

#include "common.h"
/**
 * @file CUDAImageProcessing.h
 * @brief Header file for CUDA-based image processing functions, including white balance, gamma correction, and image cropping.
 */

/**
 * @brief Applies white balance and gamma correction to an image stored on the GPU.
 *
 * This function adjusts the white balance and applies gamma correction to an input image
 * represented as a `cv::cuda::GpuMat` with 8-bit 3-channel (RGB) data.
 *
 * @param[in,out] gpuImage The input image on the GPU (CV_8UC3). The white balance and gamma corrections
 *                         are applied in-place.
 * @param[in] gamma The gamma correction factor. A value less than 1 darkens the image, while a value
 *                  greater than 1 brightens it.
 *
 * @note This function assumes that the input image is already in RGB format.
 * @note The input image must be stored in GPU memory and represented as a `cv::cuda::GpuMat`.
 */
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float gamma);

/**
 * @brief Crops and reorganizes a 16-bit Bayer image stored on the GPU.
 *
 * This function crops and reorganizes a Bayer image stored in a single-channel 16-bit format.
 * It processes even and odd rows differently to retain only the meaningful pixel data according
 * to a specific Bayer pattern.
 *
 * @param[in] src Pointer to the input image data in GPU memory. The source image is in Bayer format,
 *                stored as a single channel (16-bit per pixel).
 * @param[out] dst Pointer to the output image data in GPU memory. The reorganized image data
 *                 will be written to this location.
 * @param[in] srcWidth The width (number of columns) of the input image.
 * @param[in] dstWidth The width (number of columns) of the output image.
 * @param[in] height The height (number of rows) of both the input and output images.
 *
 * @note The memory for `dst` must be allocated before calling this function.
 * @note Both `src` and `dst` must point to memory on the GPU.
 */
void cropAndReorganizeImageCUDA(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height);


/**
 * @brief Converts a BGR image to YUV420P format on the GPU.
 *
 * This function converts a `cv::cuda::GpuMat` containing a BGR image to a YUV420P format image.
 * The input image is resized if its resolution exceeds Full HD (1920x1080).
 * The conversion is performed entirely on the GPU using a CUDA kernel.
 *
 * @param[in] bgrImage The input image on the GPU (CV_8UC3). This must be a BGR-formatted image.
 * @param[out] yuvImage The output image on the GPU (CV_8UC1). The resulting image is in YUV420P format,
 *                      with the Y channel followed by subsampled U and V channels.
 *
 * @details
 * The YUV420P format consists of a full-resolution Y channel and subsampled U and V channels
 * (each with 1/4th the number of pixels of the Y channel). This format is widely used in video
 * encoding and processing.
 *
 * - **Y Channel**: The luminance channel, stored at the start of the output image.
 * - **U Channel**: The chroma channel (blue projection), stored after the Y channel.
 * - **V Channel**: The chroma channel (red projection), stored after the U channel.
 *
 * The CUDA kernel operates on 2D grids of threads, processing the Y channel for every pixel
 * and the U/V channels for every 2x2 block of pixels in the input BGR image.
 *
 * @note
 * - The input and output images must reside in GPU memory and be represented as `cv::cuda::GpuMat`.
 * - The input image is resized to 1920x1080 if its resolution exceeds Full HD.
 * - The output image's memory is automatically allocated and should not be pre-initialized.
 *
 * @warning
 * - Ensure that the input image is in BGR format. Incorrect formats may lead to undefined behavior.
 * - Use CUDA synchronization functions (e.g., `cudaDeviceSynchronize()`) to ensure kernel completion.
 *
 * Example usage:
 * @code
 * cv::cuda::GpuMat bgrImage = cv::cuda::imread("input.jpg", cv::IMREAD_COLOR);
 * cv::cuda::GpuMat yuvImage;
 * BGRtoYUV420P(bgrImage, yuvImage);
 * cv::Mat output;
 * yuvImage.download(output);
 * cv::imwrite("output.yuv", output);
 * @endcode
 */
void BGRtoYUV420P(const cv::cuda::GpuMat& bgrImage, cv::cuda::GpuMat& yuvImage);


#endif // CUDA_IMAGE_PROCESSING_H

