/*
Kyra Riedel
April 23, 2026
Scores and ranks ink combinations.
*/

#include "scoring.h"

#include "image_utils.h"
#include "riso_output.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {

double computePerceptualLabSimilarity(const cv::Mat& sourceBgr, const cv::Mat& compositeBgr) {
    cv::Mat srcLab, cmpLab;
    cv::cvtColor(sourceBgr, srcLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(compositeBgr, cmpLab, cv::COLOR_BGR2Lab);

    const int totalPixels = srcLab.rows * srcLab.cols;
    if (totalPixels <= 0) {
        return 0.0;
    }
    double sumDeltaE = 0.0;
    for (int y = 0; y < srcLab.rows; ++y) {
        const cv::Vec3b* sRow = srcLab.ptr<cv::Vec3b>(y);
        const cv::Vec3b* cRow = cmpLab.ptr<cv::Vec3b>(y);
        for (int x = 0; x < srcLab.cols; ++x) {
            const double l1 = static_cast<double>(sRow[x][0]) * (100.0 / 255.0);
            const double a1 = static_cast<double>(sRow[x][1]) - 128.0;
            const double b1 = static_cast<double>(sRow[x][2]) - 128.0;
            const double l2 = static_cast<double>(cRow[x][0]) * (100.0 / 255.0);
            const double a2 = static_cast<double>(cRow[x][1]) - 128.0;
            const double b2 = static_cast<double>(cRow[x][2]) - 128.0;
            const double dl = l1 - l2;
            const double da = a1 - a2;
            const double db = b1 - b2;
            sumDeltaE += std::sqrt(dl * dl + da * da + db * db);
        }
    }
    const double meanDeltaE = sumDeltaE / static_cast<double>(totalPixels);
    return std::clamp(1.0 - (meanDeltaE / 100.0), 0.0, 1.0);
}

double computePixelRmseSimilarity(const cv::Mat& sourceBgr, const cv::Mat& compositeBgr) {
    cv::Mat src32f, cmp32f, diff;
    sourceBgr.convertTo(src32f, CV_32F);
    compositeBgr.convertTo(cmp32f, CV_32F);
    cv::absdiff(src32f, cmp32f, diff);
    diff = diff.mul(diff);

    const cv::Scalar msePerCh = cv::mean(diff);
    const double mse = (msePerCh[0] + msePerCh[1] + msePerCh[2]) / 3.0;
    const double rmse = std::sqrt(std::max(0.0, mse));
    return std::clamp(1.0 - (rmse / 255.0), 0.0, 1.0);
}

double computeEdgeStructureSimilarity(const cv::Mat& sourceBgr, const cv::Mat& compositeBgr) {
    cv::Mat srcGray, cmpGray;
    cv::cvtColor(sourceBgr, srcGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(compositeBgr, cmpGray, cv::COLOR_BGR2GRAY);

    cv::Mat src32f, cmp32f;
    srcGray.convertTo(src32f, CV_32F, 1.0 / 255.0);
    cmpGray.convertTo(cmp32f, CV_32F, 1.0 / 255.0);

    cv::Mat sx1, sy1, sx2, sy2;
    cv::Sobel(src32f, sx1, CV_32F, 1, 0, 3);
    cv::Sobel(src32f, sy1, CV_32F, 0, 1, 3);
    cv::Sobel(cmp32f, sx2, CV_32F, 1, 0, 3);
    cv::Sobel(cmp32f, sy2, CV_32F, 0, 1, 3);

    cv::Mat g1, g2;
    cv::magnitude(sx1, sy1, g1);
    cv::magnitude(sx2, sy2, g2);

    cv::Mat absDiff;
    cv::absdiff(g1, g2, absDiff);
    const double meanAbsDiff = cv::mean(absDiff)[0];

    cv::Mat maxGrad;
    cv::max(g1, g2, maxGrad);
    const double norm = cv::mean(maxGrad)[0] + 1e-6;
    return std::clamp(1.0 - (meanAbsDiff / norm), 0.0, 1.0);
}

double scoreCombination(const cv::Mat& sourceBgr, const cv::Mat& compositeBgr) {
    if (sourceBgr.empty() || compositeBgr.empty()) {
        return 0.0;
    }
    const double perceptual = computePerceptualLabSimilarity(sourceBgr, compositeBgr);
    const double pixel = computePixelRmseSimilarity(sourceBgr, compositeBgr);
    const double edge = computeEdgeStructureSimilarity(sourceBgr, compositeBgr);

    constexpr double kPerceptualWeight = 0.55;
    constexpr double kPixelWeight = 0.30;
    constexpr double kEdgeWeight = 0.15;
    const double score01 = kPerceptualWeight * perceptual + kPixelWeight * pixel + kEdgeWeight * edge;
    return std::clamp(score01 * 100.0, 0.0, 100.0);
}

void enumerateCombinationsRecursive(
    const std::vector<RisoInk>& inks,
    int layerCount,
    int start,
    std::vector<RisoInk>& current,
    std::vector<std::vector<RisoInk>>& all) {
    if (static_cast<int>(current.size()) == layerCount) {
        all.push_back(current);
        return;
    }

    for (int i = start; i <= static_cast<int>(inks.size()) - (layerCount - static_cast<int>(current.size())); ++i) {
        current.push_back(inks[i]);
        enumerateCombinationsRecursive(inks, layerCount, i + 1, current, all);
        current.pop_back();
    }
}

std::vector<std::vector<RisoInk>> enumerateCombinations(
    const std::vector<RisoInk>& selectedInks,
    int layerCount) {
    std::vector<std::vector<RisoInk>> combos;
    std::vector<RisoInk> current;
    enumerateCombinationsRecursive(selectedInks, layerCount, 0, current, combos);
    return combos;
}

}  // namespace

std::size_t countInkCombinations(std::size_t numSelectedInks, int layerCount) {
    const int n = static_cast<int>(numSelectedInks);
    const int k = layerCount;
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    int kk = std::min(k, n - k);
    std::size_t res = 1;
    for (int i = 1; i <= kk; ++i) {
        res = res * static_cast<std::size_t>(n - kk + i) / static_cast<std::size_t>(i);
    }
    return res;
}

std::vector<ScoredCombination> evaluateInkCombinations(
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& selectedInks,
    int layerCount,
    int maxSamples) {
    if (layerCount < 1 || layerCount > 6) {
        throw std::invalid_argument("Layer count must be in [1, 6].");
    }
    if (layerCount > static_cast<int>(selectedInks.size())) {
        throw std::invalid_argument("Layer count cannot exceed number of selected inks.");
    }

    std::vector<std::vector<RisoInk>> combinations = enumerateCombinations(selectedInks, layerCount);
    if (combinations.empty()) {
        return {};
    }

    if (static_cast<int>(combinations.size()) > maxSamples) {
        std::mt19937 rng(std::random_device{}());
        std::shuffle(combinations.begin(), combinations.end(), rng);
        combinations.resize(maxSamples);
    }

    std::vector<ScoredCombination> scored;
    scored.reserve(combinations.size());

    for (const auto& combo : combinations) {
        ScoredCombination item;
        item.inks = combo;
        item.composite = riso_output::buildDitheredRisoComposite(sourceBgr, combo);
        item.scorePercent = scoreCombination(sourceBgr, item.composite);
        scored.push_back(std::move(item));
    }

    std::sort(scored.begin(), scored.end(), [](const ScoredCombination& a, const ScoredCombination& b) {
        return a.scorePercent > b.scorePercent;
    });

    return scored;
}
