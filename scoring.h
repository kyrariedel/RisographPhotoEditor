/*
Kyra Riedel
April 23, 2026
Scoring data types and APIs.
*/

#pragma once

#include "riso_inks.h"

#include <opencv2/core.hpp>

#include <vector>

struct ScoredCombination {
    std::vector<RisoInk> inks;
    double scorePercent = 0.0;
    cv::Mat composite;
};

std::vector<ScoredCombination> evaluateInkCombinations(
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& selectedInks,
    int layerCount,
    int maxSamples = 200);

// C(n, k) with n = selectedInks.size(), k = layerCount. Used to detect a single forced combo.
std::size_t countInkCombinations(std::size_t numSelectedInks, int layerCount);
