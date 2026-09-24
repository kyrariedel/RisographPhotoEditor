/*
Kyra Riedel
April 23, 2026
Declarations for image utilities.
*/

#pragma once

#include "riso_inks.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace image_utils {

cv::Mat loadImageBgr(const std::string& imagePath);
cv::Mat reconstructComposite(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks);
std::vector<int> assignNearestInkPerPixel(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks);

bool saveImage(const std::string& outputPath, const cv::Mat& image);
bool ensureDirectory(const std::string& dirPath);

}  // namespace image_utils
