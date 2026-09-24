/*
Kyra Riedel
April 23, 2026
Public API for RISO output routines.
*/

#pragma once

#include "riso_inks.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace riso_output {

void computeCmykPlanes(const cv::Mat& bgr, cv::Mat& outC, cv::Mat& outM, cv::Mat& outY, cv::Mat& outK);
cv::Mat extractCMYKChannel(const cv::Mat& bgr, int channelIndex);  // 0=C, 1=M, 2=Y, 3=K
cv::Mat extractRGBChannel(const cv::Mat& bgr, int channelIndex);   // BGR order: 0=B, 1=G, 2=R

cv::Mat atkinsonDither(const cv::Mat& grayU8);

cv::Mat buildDitheredRisoComposite(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks);

cv::Mat buildDitheredRisoCompositeDepthSplit(
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks,
    const cv::Mat& depthU8,
    int nearThreshold,
    bool multiInkIsNearRegion,
    int monoInkIndexInList);

cv::Mat buildDitheredRisoCompositeFaceSplit(
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks);

std::string makeInkComboSlug(const std::vector<RisoInk>& inks);

struct SeparationExportResult {
    cv::Mat composite;
    std::vector<std::string> spotPaths;
};


SeparationExportResult exportSeparationPackage(
    const std::string& separationDir,
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks);

bool savePrintTogetherPreview(
    const std::string& outputPath,
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks,
    const cv::Mat& compositeBgr,
    int cellPx = 180);

} 
