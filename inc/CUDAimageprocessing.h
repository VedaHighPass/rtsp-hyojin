#ifndef CUDA_IMAGE_PROCESSING_H
#define CUDA_IMAGE_PROCESSING_H

#include "common.h"

/**
 * @file CUDAImageProcessing.h
 * @brief Header file for CUDA-based image processing functions, including white balance, gamma correction,
 *        image cropping, and BGR to YUV420P conversion.
 */

/**
 * @brief Applies white balance and gamma correction to an image stored on the GPU.
 *
 * Adjusts the white balance and applies gamma correction to an input image stored as a `cv::cuda::GpuMat`.
 * The processing is performed in-place.
 *
 * @param[in,out] gpuImage Input/output image on the GPU (CV_8UC3 format).
 * @param[in] gamma Gamma correction factor. Values < 1 darken the image, values > 1 brighten the image.
 *
 * @note Assumes the input image is in RGB format.
 */
void applyWhiteBalanceAndGammaCUDA(cv::cuda::GpuMat& gpuImage, float gamma);

/**
 * @brief Crops and reorganizes a 16-bit Bayer image stored on the GPU.
 *
 * This function processes even and odd rows differently according to a specific Bayer pattern.
 * The result is stored in a reorganized format.
 *
 * @param[in] src Input Bayer image data in GPU memory (single-channel, 16-bit per pixel).
 * @param[out] dst Output image data in GPU memory.
 * @param[in] srcWidth Width of the input image.
 * @param[in] dstWidth Width of the output image.
 * @param[in] height Height of both input and output images.
 *
 * @note Ensure memory for `dst` is allocated before calling this function.
 */
void cropAndReorganizeImageCUDA(const uint16_t* src, uint16_t* dst, int srcWidth, int dstWidth, int height);

/**
 * @brief Converts a BGR image to YUV420P format on the GPU.
 *
 * Converts a `cv::cuda::GpuMat` with BGR image data to YUV420P format. If the resolution exceeds Full HD
 * (1920x1080), the input image is resized.
 *
 * @param[in] bgrImage Input BGR image on the GPU (CV_8UC3 format).
 * @param[out] yuvImage Output YUV420P image on the GPU (CV_8UC1 format).
 *
 * @details
 * The YUV420P format consists of:
 * - **Y Channel**: Full-resolution luminance.
 * - **U/V Channels**: Subsampled chroma channels (1/4th resolution of Y channel).
 *
 * Processing:
 * - Y channel is computed for every pixel.
 * - U and V channels are subsampled from 2x2 blocks in the input BGR image.
 *
 * Example usage:
 * @code
 * cv::cuda::GpuMat bgrImage, yuvImage;
 * BGRtoYUV420P(bgrImage, yuvImage);
 * cv::Mat output;
 * yuvImage.download(output);
 * cv::imwrite("output.yuv", output);
 * @endcode
 */
void BGRtoYUV420P(const cv::cuda::GpuMat& bgrImage, cv::cuda::GpuMat& yuvImage);

#endif // CUDA_IMAGE_PROCESSING_H

