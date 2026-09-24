/*
Kyra Riedel
April 23, 2026
Defines the RISO palette.
*/

#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct RisoInk {
    std::string name;
    cv::Vec3b bgr;
};

inline std::vector<RisoInk> getRisoPalette() {
    // Six inks only. RGB (user-facing) -> OpenCV BGR.
    // Red/Yellow/Pink/Blue/Green match prior RISO palette values.
    return {
        {"Red", cv::Vec3b(96, 80, 241)},       // RGB (241, 80, 96)
        {"Yellow", cv::Vec3b(0, 232, 255)},   // RGB (255, 232, 0)
        {"Pink", cv::Vec3b(176, 72, 255)},    // RGB (255, 72, 176)
        {"Blue", cv::Vec3b(191, 120, 0)},     // RGB (0, 120, 191)
        {"Black", cv::Vec3b(0, 0, 0)},        // RGB (0, 0, 0)
        {"Green", cv::Vec3b(92, 169, 0)},     // RGB (0, 169, 92)
    };
}
