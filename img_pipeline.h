/*
Kyra Riedel
April 23, 2026
Declaration for interactive image pipeline.
*/

#pragma once

#include <opencv2/core.hpp>

#include <string>

// Project 1-style image filters (no video, no snowfall). Blocks until user presses 'q'.
// sourceImagePath: used as the OpenCV window title (basename only). May be empty.
// Returns the last displayed BGR image (clone), suitable as input to RISO scoring/export.
cv::Mat runInteractiveImagePipeline(const cv::Mat& imageBgr, const std::string& sourceImagePath = {});
