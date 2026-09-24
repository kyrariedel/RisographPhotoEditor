/*
Kyra Riedel and Abigail Valladolid
January 20, 2026
.h file for filter.cpp
*/
#pragma once
#include <opencv2/opencv.hpp>

void openCVGray(const cv::Mat &src, cv::Mat &dst);
void customGray(const cv::Mat &src, cv::Mat &dst);
void sepia(const cv::Mat &src, cv::Mat &dst);
int blur5x5_1( cv::Mat &src, cv::Mat &dst );
int blur5x5_2( cv::Mat &src, cv::Mat &dst );
int sobelX3x3( cv::Mat &src, cv::Mat &dst );
int sobelY3x3( cv::Mat &src, cv::Mat &dst );
int depthEstimation(cv::Mat &src, cv::Mat &dst);
int addFogDepth(cv::Mat &src, cv::Mat &dst);
int gradientMagnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels);
int snowfallBackground(cv::Mat &src, cv::Mat &dst);

// Raw DA2 depth map (CV_8UC1, same size as src). Higher values ≈ nearer camera.
int computeDepthMapU8(cv::Mat& src, cv::Mat& outDepthU8);

// Sobel gradient edges: where magnitude exceeds threshold, paint edgeBgr (or black if useBlackEdges).
void sobelEdgeHighlight(
    const cv::Mat& src,
    cv::Mat& dst,
    int magThreshold,
    const cv::Vec3b& edgeBgr,
    bool useBlackEdges);
int colorfulFaces(cv::Mat &src, cv::Mat &dst);
int emboss(cv::Mat &src, cv::Mat &dst);
int motionTrail(cv::Mat &frame, cv::Mat &prevGray, 
                cv::Mat &trail, cv::Mat &dst, 
                float decay = 0.9, int thresholdVal = 25);
void adjustBrightnessContrast(const cv::Mat &src, cv::Mat &dst, float brightness, float contrast);