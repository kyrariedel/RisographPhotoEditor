/*
Kyra Riedel and Abigail Valladolid
January 20, 2026
Filter file to implement various image filters for images and live video
*/
#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include "DA2Network.hpp"
#include "faceDetect.h"

namespace {

std::string resolveDa2ModelPath() {
    namespace fs = std::filesystem;
    if (const char* e = std::getenv("RISO_DA2_MODEL")) {
        if (e[0] != '\0' && fs::exists(e)) {
            return std::string(e);
        }
    }
    static constexpr const char* kCandidates[] = {
        "./model_fp16.onnx",
        "../model_fp16.onnx",
        "../../model_fp16.onnx",
    };
    for (const char* p : kCandidates) {
        if (fs::exists(p)) {
            return std::string(p);
        }
    }
    return {};
}

void printDa2ModelHint() {
    std::cerr << "Depth model not found. Set RISO_DA2_MODEL to your Depth-Anything V2 ONNX file, or place "
                 "model_fp16.onnx in the current directory or the parent directory (typical when running from "
                 "build/).\n";
}

}  // namespace

// Q3: openCV grayscale
void openCVGray(const cv::Mat &src, cv::Mat &dst) {
    cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY); // https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_gray
}

// Q4: inverted red channel grayscale
void customGray(const cv::Mat &src, cv::Mat &dst) {
    // Ensure source is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return;
    }
    dst = src.clone();
    // loop through each pixel and subtract red channel from 255, then copy the value to all three color channels
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            cv::Vec3b &p = dst.at<cv::Vec3b>(y, x);
            uchar newP = 255 - p[2];
            p[0] = newP;
            p[1] = newP;
            p[2] = newP;
        }
    }
}

// Q5: sepia (antique)
void sepia(const cv::Mat &src, cv::Mat &dst) {
    // Ensure source is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return;
    }
    dst = src.clone();
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            // get original pixel values
            cv::Vec3b orig = src.at<cv::Vec3b>(y, x);
            uchar b = orig[0], g = orig[1], r = orig[2]; // bgr order

            // modify each channel by sepia matrix
            int newB = (int)(0.272 * r + 0.534 * g + 0.131 * b);
            int newG = (int)(0.349 * r + 0.686 * g + 0.168 * b); 
            int newR = (int)(0.393 * r + 0.769 * g + 0.189 * b);

            // check if values exceed 255; if so, set to 255
            if (newB > 255) newB = 255;
            if (newG > 255) newG = 255;
            if (newR > 255) newR = 255;

            cv::Vec3b &p = dst.at<cv::Vec3b>(y, x);
            // convert from int to uchar (32->8bit)
            p[0] = (uchar)newB;
            p[1] = (uchar)newG;
            p[2] = (uchar)newR;
        }
    }
}

/* 6A: Blurs each color channel separately using the integer approximation of 
        a Gaussian [1 2 4 2 1; 2 4 8 4 2; 4 8 16 8 4; 2 4 8 4 2; 1 2 4 2 1]; does not modify src img.
*/
int blur5x5_1( cv::Mat &src, cv::Mat &dst ) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    dst = src.clone();

    // gaussian kernel
    int kernel[5][5] = {
        {1, 2, 4, 2, 1},
        {2, 4, 8, 4, 2},
        {4, 8, 16, 8, 4},
        {2, 4, 8, 4, 2},
        {1, 2, 4, 2, 1}
    };
    int kernelSum = 100; // sum of all kernel values

    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            int sumB = 0, sumG = 0, sumR = 0;
            // apply kernel
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    cv::Vec3b p = src.at<cv::Vec3b>(y + ky, x + kx);
                    int kVal = kernel[ky + 2][kx + 2];
                    sumB += p[0] * kVal;
                    sumG += p[1] * kVal;
                    sumR += p[2] * kVal;
                }
            }
            cv::Vec3b &dstP = dst.at<cv::Vec3b>(y, x);
            // convert sum of channel / total sum from int to uchar (32->8bit)
            dstP[0] = (uchar)(sumB / kernelSum);
            dstP[1] = (uchar)(sumG / kernelSum);
            dstP[2] = (uchar)(sumR / kernelSum);
        }
    }
    return 0;
}

/* 6B: Faster 5x5 blur filter using separable 1x5 filters ([1 2 4 2 1] vertical and horizontal); 
        only works on color images
*/
int blur5x5_2( cv::Mat &src, cv::Mat &dst ) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    dst = src.clone();

    int kernel[5] = {1, 2, 4, 2, 1};
    int kernelSum = 10; // sum of 1x5 kernel values
    cv::Mat temp = src.clone();

    // for seperable filters, do one horizontal and one vertical pass

    // horizontal pass
    for (int y = 0; y < src.rows; y++) {
        for (int x = 2; x < src.cols - 2; x++) {
            int sumB = 0, sumG = 0, sumR = 0;
            for (int kx = -2; kx <= 2; kx++) {
                cv::Vec3b p = src.ptr<cv::Vec3b>(y)[x + kx];
                int k = kernel[kx + 2];
                sumB += p[0] * k;
                sumG += p[1] * k;
                sumR += p[2] * k;
            }
            cv::Vec3b &tempP = temp.ptr<cv::Vec3b>(y)[x];
            // convert sum of channel / total sum from int to uchar (32->8bit)
            tempP[0] = (uchar)(sumB / kernelSum);
            tempP[1] = (uchar)(sumG / kernelSum);
            tempP[2] = (uchar)(sumR / kernelSum);
        }
    }
    // vertical pass - same as horizontal but y and x swapped
    for (int y = 2; y < src.rows - 2; y++) {
        for (int x = 0; x < src.cols; x++) {
            int sumB = 0, sumG = 0, sumR = 0;
            for (int ky = -2; ky <= 2; ky++) {
                cv::Vec3b p = temp.ptr<cv::Vec3b>(y + ky)[x];
                int k = kernel[ky + 2];
                sumB += p[0] * k;
                sumG += p[1] * k;
                sumR += p[2] * k;
            }
            cv::Vec3b &dstP = dst.ptr<cv::Vec3b>(y)[x];
            dstP[0] = (uchar)(sumB / kernelSum);
            dstP[1] = (uchar)(sumG / kernelSum);
            dstP[2] = (uchar)(sumR / kernelSum);
        }
    }
    return 0;
}

/* Q7: 
https://en.wikipedia.org/wiki/Sobel_operator
Sobel 3x3 matrices taken from wiki page above. The X filter should be positive right. 
Both the input and output images should be color images, but the output image needs to be of type 16SC3 (signed short) because the values can be in the range [-255, 255].
*/

int sobelX3x3( cv::Mat &src, cv::Mat &dst ) {
    // check src is color
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    
    // 16SC3 (signed short) output
    dst.create(src.rows, src.cols, CV_16SC3);
    
    // X kernel - positive right
    int kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    // loop through image, skipping borders
    for (int y = 1; y < src.rows - 1; y++) {
        for (int x = 1; x < src.cols - 1; x++) {
            // loop through each channel
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                // apply kernel
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        uchar p = src.at<cv::Vec3b>(y + ky, x + kx)[c];
                        int k = kernel[ky + 1][kx + 1];
                        sum += p * k;
                    }
                }
                // store signed val
                dst.at<cv::Vec3s>(y, x)[c] = (short)sum;
            }
        }
    }
    
    // set border pixels to 0
    for (int x = 0; x < src.cols; x++) {
        dst.at<cv::Vec3s>(0, x) = cv::Vec3s(0, 0, 0);
        dst.at<cv::Vec3s>(src.rows - 1, x) = cv::Vec3s(0, 0, 0);
    }
    for (int y = 0; y < src.rows; y++) {
        dst.at<cv::Vec3s>(y, 0) = cv::Vec3s(0, 0, 0);
        dst.at<cv::Vec3s>(y, src.cols - 1) = cv::Vec3s(0, 0, 0);
    }
    
    return 0;
}

// sobel Y 3x3 filter for grayscale images; same as sobelX3x3 but with Y kernel and grayscale input/output
int sobelY3x3( cv::Mat &src, cv::Mat &dst ) {
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    
    // 16SC3 (signed short) output
    dst.create(src.rows, src.cols, CV_16SC3);

    // Y kernel - positive up
    int kernel[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };

// loop through image, skipping borders
for (int y = 1; y < src.rows - 1; y++) {
    for (int x = 1; x < src.cols - 1; x++) {
        // loop through each channel
        for (int c = 0; c < 3; c++) {
            int sum = 0;
            // apply kernel
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    uchar p = src.at<cv::Vec3b>(y + ky, x + kx)[c];
                    int k = kernel[ky + 1][kx + 1];
                    sum += p * k;
                }
            }
            // store signed val
            dst.at<cv::Vec3s>(y, x)[c] = (short)sum;
        }
    }
}

// set border pixels to 0
for (int x = 0; x < src.cols; x++) {
    dst.at<cv::Vec3s>(0, x) = cv::Vec3s(0, 0, 0);
    dst.at<cv::Vec3s>(src.rows - 1, x) = cv::Vec3s(0, 0, 0);
}
for (int y = 0; y < src.rows; y++) {
    dst.at<cv::Vec3s>(y, 0) = cv::Vec3s(0, 0, 0);
    dst.at<cv::Vec3s>(y, src.cols - 1) = cv::Vec3s(0, 0, 0);
}

return 0;
}

// Q8: Implement a function that generates a gradient magnitude image from the X and Y Sobel images
int gradientMagnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst){

    // Check that inputs are 3 channel signed short images
    if (sx.type() != CV_16SC3 || sy.type() != CV_16SC3) {
        printf("Error: magnitude expects CV_16SC3 images\n");
        return -1;
    }

    // create destination as an unchar color image
    dst.create(sx.rows, sx.cols, CV_8UC3);

    // loop through each image (minus the borders)
    for (int y = 1; y <sx.rows - 1; y++) {
        for (int x = 1; x < sx.cols - 1; x++){
            cv::Vec3s gx = sx.at<cv::Vec3s>(y,x);
            cv::Vec3s gy = sy.at<cv::Vec3s>(y,x);
            cv::Vec3b out;

            // get magnitude per channel
            for (int c = 0; c < 3; c++) {
                int sx_val = gx[c];
                int sy_val = gy[c];

                int mag = static_cast<int>(
                    std::sqrt(sx_val * sx_val + sy_val * sy_val)
                    );

                if (mag > 255) mag = 255;

                out[c] = static_cast<unsigned char> (mag);
                
            }
            dst.at<cv::Vec3b>(y, x) = out;
        }
    }

    // place zeros on borders
    for (int x = 0; x < sx.cols; x++) {
        dst.at<cv::Vec3b>(0, x) = cv::Vec3b(0, 0, 0);
        dst.at<cv::Vec3b>(sx.rows - 1, x) = cv::Vec3b(0, 0, 0);
    }
    for (int y = 0; y < sx.rows; y++) {
        dst.at<cv::Vec3b>(y, 0) = cv::Vec3b(0, 0, 0);
        dst.at<cv::Vec3b>(y, sx.cols - 1) = cv::Vec3b(0, 0, 0);
    }

    return 0;
}

// Q9: Implement a function that blurs and quantizes a color image
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {

    // check levels
    if (levels <= 0) levels = 1;

    // create destination image, ensure it its the same size as src
    dst.create(src.size(), src.type());

    // blur the input using 5x5 blur function
    cv::Mat temp;
    blur5x5_1(src, temp);

    // compute the size of each quantization bucket
    int bucketSize = 255 / levels;

    // loop through each pixel

    for (int y = 0; y < temp.rows; y++) {
        for (int x = 0; x < temp.cols; x++) {

            // get original color pixel
            cv::Vec3b color = temp.at<cv::Vec3b>(y, x);
            cv::Vec3b quantized;

            // quantize each channel 
            for (int c = 0; c < 3; c++) {
                int val = color[c];          
                int level = val / bucketSize;  
                int newVal = level * bucketSize;
                if (newVal > 255) newVal = 255; 
                quantized[c] = static_cast<unsigned char>(newVal);
            }

            // assign quantized pixel to output image
            dst.at<cv::Vec3b>(y, x) = quantized;
        }
    }
    return 0;
}



// modified from da2 example file; displays depth live video
int depthEstimation(cv::Mat &src, cv::Mat &dst) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    
    
    // make a DA2Network object static so it only loads once
    static DA2Network* da_net = nullptr;
    static bool initialized = false;
    
    static bool loadFailed = false;
    if (loadFailed) {
        dst = src.clone();
        return -1;
    }

    // initialize network
    if (!initialized) {
        const std::string modelPath = resolveDa2ModelPath();
        if (modelPath.empty()) {
            printDa2ModelHint();
            loadFailed = true;
            dst = src.clone();
            return -1;
        }
        try {
            da_net = new DA2Network(modelPath.c_str());
            initialized = true;
        } catch (...) {
            std::cerr << "Error loading DA2Network (path: " << modelPath << ")\n";
            loadFailed = true;
            dst = src.clone();
            return -1;
        }
    }

    // check if network is valid
    if (da_net == nullptr) {
        printf("Error: DA2Network is null\n");
        dst = src.clone();
        return -1;
    }

    // scale the network input based on image size to improve lag (kinda)
    cv::Mat scaled;
    cv::resize(src, scaled, cv::Size(), 0.5, 0.5);
    
    float scale_factor = 256.0 / (scaled.rows);
    
    // set up the network input
    try {
        if (da_net->set_input(scaled, scale_factor) != 0) {
            printf("Error: Failed to set input for depth network\n");
            dst = src.clone();
            return -1;
        }
        
        // run the network
        cv::Mat output;
        if (da_net->run_network(output, scaled.size()) != 0) {
            printf("Error: Failed to run depth network\n");
            dst = src.clone();
            return -1;
        }
    
        // apply a color map to the depth outputs
        cv::Mat dst_vis;
        cv::applyColorMap(output, dst_vis, cv::COLORMAP_INFERNO);
        
        // resize back to original size
        cv::resize(dst_vis, dst, src.size());
    } catch (...) {
        printf("Error: Exception during depth estimation\n");
        dst = src.clone();
        return -1;
    }
    
    return 0;
}

int computeDepthMapU8(cv::Mat& src, cv::Mat& outDepthU8) {
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        outDepthU8.release();
        return -1;
    }

    static DA2Network* da_net = nullptr;
    static bool initialized = false;
    static bool loadFailed = false;

    if (loadFailed) {
        outDepthU8.release();
        return -1;
    }

    if (!initialized) {
        const std::string modelPath = resolveDa2ModelPath();
        if (modelPath.empty()) {
            printDa2ModelHint();
            loadFailed = true;
            outDepthU8.release();
            return -1;
        }
        try {
            da_net = new DA2Network(modelPath.c_str());
            initialized = true;
        } catch (...) {
            std::cerr << "Error loading DA2Network (path: " << modelPath << ")\n";
            loadFailed = true;
            outDepthU8.release();
            return -1;
        }
    }
    if (da_net == nullptr) {
        printf("Error: DA2Network is null\n");
        outDepthU8.release();
        return -1;
    }

    cv::Mat scaled;
    cv::resize(src, scaled, cv::Size(), 0.5, 0.5);
    float scale_factor = 256.0f / static_cast<float>(scaled.rows);

    cv::Mat output;
    try {
        if (da_net->set_input(scaled, scale_factor) != 0) {
            printf("Unable to set input\n");
            outDepthU8.release();
            return -1;
        }

        if (da_net->run_network(output, scaled.size()) != 0) {
            printf("Unable to run network\n");
            outDepthU8.release();
            return -1;
        }
    } catch (...) {
        printf("Unable to create depth map\n");
        outDepthU8.release();
        return -1;
    }

    cv::resize(output, outDepthU8, src.size());
    return 0;
}

// add fog to the image using depth values with exponential fog effect (12)
int addFogDepth(cv::Mat &src, cv::Mat &dst) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }

    cv::Mat depth_map;
    if (computeDepthMapU8(src, depth_map) != 0) {
        dst = src.clone();
        return -1;
    }

    // fog param; higher = less fog, lower = more fog
    const float density = 0.5f;
    const cv::Vec3b fogColor(200, 200, 200);
    
    dst = src.clone();
    
    // Fog formula: fog_amount = 1 - exp(-distance/fog_density) (exponential effect of distance)
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // bounds check
            if (y >= depth_map.rows || x >= depth_map.cols) {
                continue;
            }
            
            // get depth
            float depth_value = static_cast<float>(depth_map.at<uchar>(y, x));
            
            // invert to get proper depth effect
            float dist = (255.0f - depth_value) / 255.0f;
            
            // exponential fog amount
            float fog = 1.0f - std::exp(-dist / density);
            
            // clamp fog amount
            fog = std::max(0.0f, std::min(1.0f, fog));
            
            // blend color
            cv::Vec3b orig = src.at<cv::Vec3b>(y, x);
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);
            
            for (int c = 0; c < 3; c++) {
                float blended = orig[c] * (1.0f - fog) + fogColor[c] * fog;
                pixel[c] = static_cast<uchar>(std::max(0.0f, std::min(255.0f, blended)));
            }
        }
    }
    
    return 0;
}
// snowfall background using depth (11)
int snowfallBackground(cv::Mat &src, cv::Mat &dst) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    
    static DA2Network* da_net = nullptr;
    static bool initialized = false;
    static bool loadFailed = false;

    if (loadFailed) {
        dst = src.clone();
        return -1;
    }

    if (!initialized) {
        const std::string modelPath = resolveDa2ModelPath();
        if (modelPath.empty()) {
            printDa2ModelHint();
            loadFailed = true;
            dst = src.clone();
            return -1;
        }
        try {
            da_net = new DA2Network(modelPath.c_str());
            initialized = true;
        } catch (...) {
            std::cerr << "Unable to set up depth network (path: " << modelPath << ")\n";
            loadFailed = true;
            dst = src.clone();
            return -1;
        }
    }
    if (da_net == nullptr) {
        printf("Unable to find network\n");
        dst = src.clone();
        return -1;
    }
    
    // generate depth map
    cv::Mat scaled;
    cv::resize(src, scaled, cv::Size(), 0.5, 0.5);
    float scale_factor = 256.0 / (scaled.rows);
    
    cv::Mat depth_output;
    try {
        if (da_net->set_input(scaled, scale_factor) != 0) {
            printf("Unable to set input\n");
            dst = src.clone();
            return -1;
        }
        
        if (da_net->run_network(depth_output, scaled.size()) != 0) {
            printf("Unable to run network\n");
            dst = src.clone();
            return -1;
        }
    } catch (...) {
        printf("Unable to create depth map\n");
        dst = src.clone();
        return -1;
    }
    
    cv::Mat depth_map;
    cv::resize(depth_output, depth_map, src.size());

    if (depth_map.empty() || depth_map.size() != src.size()) {
        printf("Invalid depth map\n");
        dst = src.clone();
        return -1;
    }
    
    // snowflake w/ positions using static var
    static std::vector<cv::Point2f> snow;
    static bool snow_initialized = false;
    
    // initialize snowflakes
    if (!snow_initialized) {
        int snowNum = 400; 
        for (int i = 0; i < snowNum; i++) {
            float x = (float)(rand() % src.cols);
            float y = (float)(rand() % src.rows);
            snow.push_back(cv::Point2f(x, y));
        }
        snow_initialized = true;
    }
    
    // update positions
    for (auto& flake : snow) {
        flake.x += 2, flake.y += 2;
        // reset when reaches bottom, wrap horizontal
        if (flake.y > src.rows) {
            flake.y = 0;
            flake.x = (float)(rand() % src.cols);
        }
        if (flake.x < 0) flake.x = src.cols - 1;
        if (flake.x >= src.cols) flake.x = 0;
    }
    
    dst = src.clone();
    
    // background = blue
    cv::Vec3b bg_color(200, 150, 100);
    
    // pixels with depth < 100 = background
    int depth_threshold = 100;
    
    // replace background pixels
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            // bounds check
            if (y >= depth_map.rows || x >= depth_map.cols) {
                continue;
            }
            
            uchar depth_val = depth_map.at<uchar>(y, x);
            if (depth_val < depth_threshold) {
                dst.at<cv::Vec3b>(y, x) = bg_color;
            }
        }
    }
    
    // draw snowflakes
    for (const auto& flake : snow) {
        int x = (int)flake.x;
        int y = (int)flake.y;
        
        if (x >= 0 && x < src.cols && y >= 0 && y < src.rows) {
            // Bounds check for depth map access
            if (y < depth_map.rows && x < depth_map.cols) {
                uchar depth_val = depth_map.at<uchar>(y, x);
                
                // only drawn on background pixels
                if (depth_val < depth_threshold) {
                    int size = rand() % 2 + 1; // vary size
                    cv::circle(dst, cv::Point(x, y), size, cv::Scalar(255, 255, 255), -1);
                }
            }
        }
    }
    
    return 0;
}

// Make faces colorful while the rest of the image is greyscale (12)
int colorfulFaces(cv::Mat &src, cv::Mat &dst) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    
    // convert to greyscale, then 3-channel BGR
    cv::Mat grey;
    cv::cvtColor(src, grey, cv::COLOR_BGR2GRAY);
    cv::Mat greyBGR;
    cv::cvtColor(grey, greyBGR, cv::COLOR_GRAY2BGR);
    dst = greyBGR.clone();
    
    // detect faces
    cv::Mat greyForDetection;
    cv::cvtColor(src, greyForDetection, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    detectFaces(greyForDetection, faces);
    
    // no faces, return grayscale image
    if (faces.empty()) {
        return 0;
    }
    
    // copy face regions from original to greyscale image
    for (const auto& face : faces) {
        cv::Rect safeRect = face & cv::Rect(0, 0, src.cols, src.rows);
        src(safeRect).copyTo(dst(safeRect));
    }
    
    return 0;
}

// Make an embossing effect (hint, use the Sobel X and Sobel Y signed outputs and take a dot product with a selected direction like (0.7071, 0.7071). (12)
int emboss(cv::Mat &src, cv::Mat &dst) {
    // check src is 3-channel BGR
    if (src.channels() != 3) {
        printf("Source is not a 3-channel BGR\n");
        dst = src.clone();
        return -1;
    }
    
    // Sobel X and Y gradients, output image (8UC3)
    cv::Mat sobelX, sobelY;
    sobelX3x3(src, sobelX);
    sobelY3x3(src, sobelY);
    dst.create(src.rows, src.cols, CV_8UC3);
    
    // embossing direction vector
    const float dir_x = 0.7071f;
    const float dir_y = 0.7071f; 
    const float offset = 128.0f;  // offset to mid-gray
    
    // embossed image dot product gradient
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            cv::Vec3b &pixel = dst.at<cv::Vec3b>(y, x);
            
            for (int c = 0; c < 3; c++) {
                // get Sobel X, Y
                short sx = sobelX.at<cv::Vec3s>(y, x)[c];
                short sy = sobelY.at<cv::Vec3s>(y, x)[c];
                
                // dot product: (sx, sy) · (dir_x, dir_y)
                float embossed = sx * dir_x + sy * dir_y;
                
                // offset + clamp to [0, 255]
                float value = embossed + offset;
                value = std::max(0.0f, std::min(255.0f, value));
                
                pixel[c] = static_cast<uchar>(value);
            }
        }
    }
    
    return 0;
}


/* 
Extension 1: this captures moving objects and places a white trail behind the motion
*/
int motionTrail(
    cv::Mat &frame, cv::Mat &prevGray, cv::Mat &trail, 
    cv::Mat &dst, float decay, int thresholdVal) {
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Initialize on first frame
        if (prevGray.empty()) {
            prevGray = gray.clone();
            trail = cv::Mat::zeros(frame.size(),CV_8UC3);
            dst = frame.clone();
            return 0;
        }

        // difference between frames
        cv::Mat diff;
        cv::absdiff(gray, prevGray, diff);
        
        // motion threshold and fading the prev trail
        cv::Mat motionMask;
        cv::threshold(diff, motionMask, thresholdVal, 255, cv::THRESH_BINARY);
        cv::GaussianBlur(motionMask, motionMask, cv::Size(5,5), 0);
        trail *= decay;

        // look for new motion
        cv::Mat coloredMotion;
        cv::cvtColor(motionMask, coloredMotion, cv::COLOR_GRAY2BGR);
        trail += coloredMotion;

        // overlay trail
        cv::addWeighted(frame, 1.0, trail, 1.0, 0, dst);

        // frame update
        prevGray = gray.clone();

        return 0;

    }

// Adjust brightness and contrast of an image
// brightness: -100 to 100 (0 = no change, positive = brighter, negative = darker)
// contrast: -100 to 100 (0 = no change, positive = more contrast, negative = less contrast)
void adjustBrightnessContrast(const cv::Mat &src, cv::Mat &dst, float brightness, float contrast) {
    if (src.channels() != 3) {
        dst = src.clone();
        return;
    }
    
    // Convert brightness and contrast to scaling factors
    // Brightness: add/subtract value
    // Contrast: multiply by factor (1.0 = no change)
    float alpha = 1.0f + (contrast / 100.0f);  // Contrast: 1.0 + (-1.0 to 1.0)
    float beta = brightness;  // Brightness: direct offset
    
    // Apply formula: new_pixel = alpha * pixel + beta
    src.convertTo(dst, -1, alpha, beta);
    
    // Clamp values to [0, 255]
    cv::threshold(dst, dst, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(dst, dst, 0, 0, cv::THRESH_TOZERO);
}

void sobelEdgeHighlight(
    const cv::Mat& src,
    cv::Mat& dst,
    int magThreshold,
    const cv::Vec3b& edgeBgr,
    bool useBlackEdges) {
    if (src.channels() != 3 || src.empty()) {
        dst = src.clone();
        return;
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);

    double magMax = 0.0;
    cv::minMaxLoc(mag, nullptr, &magMax);
    if (magMax < 1e-6) {
        magMax = 1.0;
    }

    dst = src.clone();
    const cv::Vec3b pen = useBlackEdges ? cv::Vec3b(0, 0, 0) : edgeBgr;
    const float scale = static_cast<float>(255.0 / magMax);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            const float n = mag.at<float>(y, x) * scale;
            if (n >= static_cast<float>(magThreshold)) {
                dst.at<cv::Vec3b>(y, x) = pen;
            }
        }
    }
}