/*
Kyra Riedel
April 23, 2026
Image loading and utility helpers.
*/

#include "image_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace image_utils {

namespace {

double colorDistance(const cv::Vec3b& c1, const cv::Vec3b& c2) {
    const int db = static_cast<int>(c1[0]) - static_cast<int>(c2[0]);
    const int dg = static_cast<int>(c1[1]) - static_cast<int>(c2[1]);
    const int dr = static_cast<int>(c1[2]) - static_cast<int>(c2[2]);
    return std::sqrt(static_cast<double>(db * db + dg * dg + dr * dr));
}

}  // namespace

cv::Mat loadImageBgr(const std::string& imagePath) {
    cv::Mat source = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (source.empty()) {
        throw std::runtime_error("Failed to load image: " + imagePath);
    }
    return source;
}

std::vector<int> assignNearestInkPerPixel(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks) {
    if (sourceBgr.empty() || inks.empty()) {
        return {};
    }

    std::vector<int> assignments;
    assignments.reserve(static_cast<size_t>(sourceBgr.rows * sourceBgr.cols));

    for (int y = 0; y < sourceBgr.rows; ++y) {
        for (int x = 0; x < sourceBgr.cols; ++x) {
            const cv::Vec3b pixel = sourceBgr.at<cv::Vec3b>(y, x);
            int bestInkIndex = 0;
            double bestDistance = colorDistance(pixel, inks[0].bgr);

            for (int i = 1; i < static_cast<int>(inks.size()); ++i) {
                const double candidateDistance = colorDistance(pixel, inks[i].bgr);
                if (candidateDistance < bestDistance) {
                    bestDistance = candidateDistance;
                    bestInkIndex = i;
                }
            }
            assignments.push_back(bestInkIndex);
        }
    }

    return assignments;
}

cv::Mat reconstructComposite(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks) {
    cv::Mat reconstructed(sourceBgr.rows, sourceBgr.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    if (sourceBgr.empty() || inks.empty()) {
        return reconstructed;
    }

    const std::vector<int> assignments = assignNearestInkPerPixel(sourceBgr, inks);

    int idx = 0;
    for (int y = 0; y < sourceBgr.rows; ++y) {
        for (int x = 0; x < sourceBgr.cols; ++x) {
            reconstructed.at<cv::Vec3b>(y, x) = inks[assignments[idx++]].bgr;
        }
    }

    return reconstructed;
}

bool ensureDirectory(const std::string& dirPath) {
    try {
        return std::filesystem::create_directories(dirPath) || std::filesystem::exists(dirPath);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to create output directory '" << dirPath << "': " << ex.what() << '\n';
        return false;
    }
}

bool saveImage(const std::string& outputPath, const cv::Mat& image) {
    if (!cv::imwrite(outputPath, image)) {
        std::cerr << "Failed to save image: " << outputPath << '\n';
        return false;
    }
    return true;
}

}  // namespace image_utils
