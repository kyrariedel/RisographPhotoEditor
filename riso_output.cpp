/*
Kyra Riedel
April 23, 2026
Builds RISO plates and composites.
*/

#include "riso_output.h"

#include "faceDetect.h"
#include "image_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>

// ideated from: https://antiboredom.github.io/p5.riso/

namespace riso_output {

namespace {
constexpr float kRisoGamma = 1.20f;

// https://www.imatest.com/imaging/tonal-response-gamma/
std::array<uint8_t, 256> buildRisoToneLut() {
    std::array<uint8_t, 256> lut{};
    for (int v = 0; v < 256; ++v) {
        const float norm = static_cast<float>(v) / 255.f;
        const float curved = std::pow(norm, kRisoGamma);
        lut[static_cast<size_t>(v)] = static_cast<uint8_t>(
            std::clamp(std::lround(curved * 255.f), 0L, 255L));
    }
    return lut;
}

// Returns plate from BGR
cv::Mat computeContinuousTonePlate(const cv::Mat& sourceBgr, const RisoInk& ink) {
    static const std::array<uint8_t, 256> kLut = buildRisoToneLut();

    CV_Assert(!sourceBgr.empty());
    CV_Assert(sourceBgr.type() == CV_8UC3);

    const float wB = static_cast<float>(ink.bgr[0]);
    const float wG = static_cast<float>(ink.bgr[1]);
    const float wR = static_cast<float>(ink.bgr[2]);
    const float wSum = wB + wG + wR;

    const float invDenom = (wSum > 0.f) ? (1.f / (255.f * wSum)) : (1.f / (255.f * 3.f));
    const float fallbackWB = (wSum > 0.f) ? wB : 85.f;
    const float fallbackWG = (wSum > 0.f) ? wG : 85.f;
    const float fallbackWR = (wSum > 0.f) ? wR : 85.f;

    cv::Mat plate(sourceBgr.rows, sourceBgr.cols, CV_8UC1);

    for (int y = 0; y < sourceBgr.rows; ++y) {
        const cv::Vec3b* srcRow = sourceBgr.ptr<cv::Vec3b>(y);
        uint8_t* dstRow = plate.ptr<uint8_t>(y);

        for (int x = 0; x < sourceBgr.cols; ++x) {
            const cv::Vec3b& p = srcRow[x];
            float raw01 = 0.f;
            if (wSum <= 1e-5f) {
                const float y709 =
                    (0.0722f * static_cast<float>(p[0]) + 0.7152f * static_cast<float>(p[1])
                        + 0.2126f * static_cast<float>(p[2]))
                    / 255.f;
                raw01 = 1.f - std::clamp(y709, 0.f, 1.f);
            } else {
                raw01 = (static_cast<float>(p[0]) * fallbackWB + static_cast<float>(p[1]) * fallbackWG
                            + static_cast<float>(p[2]) * fallbackWR)
                    * invDenom;
                raw01 = std::clamp(raw01, 0.f, 1.f);
            }
            const uint8_t rawU8 =
                static_cast<uint8_t>(std::clamp(std::lround(raw01 * 255.f), 0L, 255L));
            dstRow[x] = kLut[rawU8];
        }
    }

    return plate;
}

bool isBlackInkSwatch(const RisoInk& ink) {
    return ink.bgr[0] == 0 && ink.bgr[1] == 0 && ink.bgr[2] == 0;
}

bool containsBlackInk(const std::vector<RisoInk>& inks) {
    for (const RisoInk& ink : inks) {
        if (isBlackInkSwatch(ink)) {
            return true;
        }
    }
    return false;
}

int findBlackInkIndex(const std::vector<RisoInk>& inks) {
    for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
        if (isBlackInkSwatch(inks[static_cast<size_t>(i)])) {
            return i;
        }
    }
    return -1;
}

// Convert a continuous-tone grayscale plate into a BGR colour layer:
//   pixel_BGR(t) = white * (1 - t)  +  ink_BGR * t,  where t = plate/255.
cv::Mat tintPlateWithInkColor(const cv::Mat& plateU8, const cv::Vec3b& inkBgr) {
    CV_Assert(plateU8.type() == CV_8UC1);

    cv::Mat out(plateU8.rows, plateU8.cols, CV_8UC3);

    for (int y = 0; y < plateU8.rows; ++y) {
        const uint8_t* srcRow = plateU8.ptr<uint8_t>(y);
        cv::Vec3b* dstRow = out.ptr<cv::Vec3b>(y);

        for (int x = 0; x < plateU8.cols; ++x) {
            const float t = static_cast<float>(srcRow[x]) / 255.f;
            cv::Vec3b& dst = dstRow[x];
            for (int c = 0; c < 3; ++c) {
                const float v = 255.f * (1.f - t) + static_cast<float>(inkBgr[c]) * t;
                dst[c] = static_cast<uint8_t>(std::clamp(std::lround(v), 0L, 255L));
            }
        }
    }

    return out;
}

// Layers N BGR layers using multiply blend in linear sRGB.
// Multiply blend (f(a,b) = a*b) to model ink printing over layers
float srgbToLinear(uint8_t u) {
    const float x = static_cast<float>(u) / 255.f;
    return (x <= 0.04045f) ? (x / 12.92f) : std::pow((x + 0.055f) / 1.055f, 2.4f);
}

uint8_t linearToSrgb(float l) {
    const float x = (l <= 0.0031308f) ? (12.92f * l) : (1.055f * std::pow(l, 1.f / 2.4f) - 0.055f);
    return static_cast<uint8_t>(std::clamp(std::lround(x * 255.f), 0L, 255L));
}

cv::Mat multiplyBlendLayers(const std::vector<cv::Mat>& layers) {
    if (layers.empty()) {
        return {};
    }
    const int h = layers[0].rows;
    const int w = layers[0].cols;
    cv::Mat out(h, w, CV_8UC3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float lb = 1.f;
            float lg = 1.f;
            float lr = 1.f;
            for (const cv::Mat& layer : layers) {
                const cv::Vec3b p = layer.at<cv::Vec3b>(y, x);
                lb *= srgbToLinear(p[0]);
                lg *= srgbToLinear(p[1]);
                lr *= srgbToLinear(p[2]);
            }
            out.at<cv::Vec3b>(y, x) = cv::Vec3b(linearToSrgb(lb), linearToSrgb(lg), linearToSrgb(lr));
        }
    }
    return out;
}

cv::Mat toBgrPanel(const cv::Mat& img) {
    if (img.channels() == 3) {
        return img.clone();
    }
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

cv::Mat addCaptionStripBelow(const cv::Mat& contentBgr, const std::string& caption, int labelHeight) {
    CV_Assert(contentBgr.type() == CV_8UC3);
    const int lh = std::max(24, labelHeight);
    cv::Mat out(contentBgr.rows + lh, contentBgr.cols, CV_8UC3);
    contentBgr.copyTo(out(cv::Rect(0, 0, contentBgr.cols, contentBgr.rows)));
    out(cv::Rect(0, contentBgr.rows, contentBgr.cols, lh)).setTo(cv::Scalar(28, 28, 28));

    const double fontScale = std::clamp(static_cast<double>(contentBgr.cols) / 720.0, 0.35, 1.35);
    int baseline = 0;
    const cv::Size ts =
        cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
    const cv::Point org((contentBgr.cols - ts.width) / 2, contentBgr.rows + (lh + ts.height) / 2 - 2);
    cv::putText(
        out,
        caption,
        org,
        cv::FONT_HERSHEY_SIMPLEX,
        fontScale,
        cv::Scalar(248, 248, 248),
        1,
        cv::LINE_AA);
    return out;
}

cv::Mat padPanelToHeight(const cv::Mat& panel, int targetH) {
    if (panel.rows == targetH) {
        return panel.clone();
    }
    cv::Mat canvas(targetH, panel.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    panel.copyTo(canvas(cv::Rect(0, 0, panel.cols, panel.rows)));
    return canvas;
}

// holds the different layers
struct PlatePair {
    cv::Mat continuous;  // uint8 grayscale, continuous tone 0–255
    cv::Mat tinted;      // BGR colour layer for final img
};

cv::Rect largestFaceSquareRoi(const cv::Size& frameSize, const std::vector<cv::Rect>& faces) {
    if (faces.empty()) {
        return {};
    }
    size_t best = 0;
    int bestArea = 0;
    for (size_t i = 0; i < faces.size(); ++i) {
        const int area = faces[i].width * faces[i].height;
        if (area > bestArea) {
            bestArea = area;
            best = i;
        }
    }
    cv::Rect f = faces[best];
    const int side = std::max(1, std::max(f.width, f.height));
    const int cx = f.x + f.width / 2;
    const int cy = f.y + f.height / 2;
    int x = cx - side / 2;
    int y = cy - side / 2;
    x = std::clamp(x, 0, std::max(0, frameSize.width - side));
    y = std::clamp(y, 0, std::max(0, frameSize.height - side));
    const int w = std::min(side, frameSize.width - x);
    const int h = std::min(side, frameSize.height - y);
    return cv::Rect(x, y, std::max(1, w), std::max(1, h));
}

int leastPresentInkIndex(const std::vector<PlatePair>& plates) {
    if (plates.empty()) {
        return -1;
    }
    int minIdx = 0;
    double minMean = cv::mean(plates[0].continuous)[0];
    for (int i = 1; i < static_cast<int>(plates.size()); ++i) {
        const double m = cv::mean(plates[static_cast<size_t>(i)].continuous)[0];
        if (m < minMean) {
            minMean = m;
            minIdx = i;
        }
    }
    return minIdx;
}

void applyChromaAwareBlackSuppression(std::vector<PlatePair>& pairs, const cv::Mat& sourceBgr, int blackIdx) {
    if (pairs.empty() || sourceBgr.empty() || blackIdx < 0 || blackIdx >= static_cast<int>(pairs.size())) {
        return;
    }
    constexpr float kStrength = 0.85f;
    constexpr float kMinScale = 0.20f;

    cv::Mat& blackPlate = pairs[static_cast<size_t>(blackIdx)].continuous;
    for (int y = 0; y < sourceBgr.rows; ++y) {
        const cv::Vec3b* srcRow = sourceBgr.ptr<cv::Vec3b>(y);
        uint8_t* blkRow = blackPlate.ptr<uint8_t>(y);
        for (int x = 0; x < sourceBgr.cols; ++x) {
            const cv::Vec3b& p = srcRow[x];
            const float b = static_cast<float>(p[0]) / 255.f;
            const float g = static_cast<float>(p[1]) / 255.f;
            const float r = static_cast<float>(p[2]) / 255.f;
            const float maxc = std::max({r, g, b});
            const float minc = std::min({r, g, b});
            const float chroma = std::clamp(maxc - minc, 0.f, 1.f);
            const float luma = 0.0722f * b + 0.7152f * g + 0.2126f * r;

            const float suppression = kStrength * chroma * (0.25f + 0.75f * std::clamp(luma, 0.f, 1.f));
            const float scale = std::clamp(1.f - suppression, kMinScale, 1.f);
            blkRow[x] = static_cast<uint8_t>(std::clamp(std::lround(static_cast<float>(blkRow[x]) * scale), 0L, 255L));
        }
    }
}

void solveNoBlackBestMixPlates(std::vector<PlatePair>& pairs, const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks) {
    if (pairs.empty() || sourceBgr.empty() || pairs.size() != inks.size()) {
        return;
    }

    constexpr float kEps = 1e-4f;
    constexpr float kRidge = 0.015f;
    constexpr int kIters = 14;

    const int n = static_cast<int>(inks.size());
    std::vector<cv::Vec3f> inkAbsorb(static_cast<size_t>(n), cv::Vec3f(0, 0, 0));
    std::vector<float> diag(static_cast<size_t>(n), 0.f);
    float maxDiag = 1e-5f;

    for (int i = 0; i < n; ++i) {
        const cv::Vec3b bgr = inks[static_cast<size_t>(i)].bgr;
        const float lb = std::max(kEps, srgbToLinear(bgr[0]));
        const float lg = std::max(kEps, srgbToLinear(bgr[1]));
        const float lr = std::max(kEps, srgbToLinear(bgr[2]));
        const cv::Vec3f k(-std::log(lb), -std::log(lg), -std::log(lr));
        inkAbsorb[static_cast<size_t>(i)] = k;
        const float d = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + kRidge;
        diag[static_cast<size_t>(i)] = d;
        maxDiag = std::max(maxDiag, d);
    }
    const float step = 0.8f / maxDiag;

    std::vector<float> t(static_cast<size_t>(n), 0.f);
    std::vector<float> grad(static_cast<size_t>(n), 0.f);

    for (int y = 0; y < sourceBgr.rows; ++y) {
        const cv::Vec3b* srcRow = sourceBgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < sourceBgr.cols; ++x) {
            const cv::Vec3b& p = srcRow[x];
            const float lb = std::max(kEps, srgbToLinear(p[0]));
            const float lg = std::max(kEps, srgbToLinear(p[1]));
            const float lr = std::max(kEps, srgbToLinear(p[2]));
            const cv::Vec3f target(-std::log(lb), -std::log(lg), -std::log(lr));

            for (int i = 0; i < n; ++i) {
                t[static_cast<size_t>(i)] = 0.f;
            }
            for (int it = 0; it < kIters; ++it) {
                cv::Vec3f pred(0, 0, 0);
                for (int i = 0; i < n; ++i) {
                    pred += inkAbsorb[static_cast<size_t>(i)] * t[static_cast<size_t>(i)];
                }
                const cv::Vec3f err = pred - target;

                for (int i = 0; i < n; ++i) {
                    const cv::Vec3f k = inkAbsorb[static_cast<size_t>(i)];
                    grad[static_cast<size_t>(i)] =
                        2.f * (k[0] * err[0] + k[1] * err[1] + k[2] * err[2]) + 2.f * kRidge * t[static_cast<size_t>(i)];
                }
                for (int i = 0; i < n; ++i) {
                    const float v = t[static_cast<size_t>(i)] - step * grad[static_cast<size_t>(i)];
                    t[static_cast<size_t>(i)] = std::clamp(v, 0.f, 1.f);
                }
            }

            for (int i = 0; i < n; ++i) {
                pairs[static_cast<size_t>(i)].continuous.at<uint8_t>(y, x) = static_cast<uint8_t>(
                    std::clamp(std::lround(t[static_cast<size_t>(i)] * 255.f), 0L, 255L));
            }
        }
    }
}

std::vector<PlatePair> buildAllPlates(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks) {
    std::vector<PlatePair> pairs;
    pairs.reserve(inks.size());
    for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
        const RisoInk& ink = inks[static_cast<size_t>(i)];
        PlatePair pp;
        pp.continuous = computeContinuousTonePlate(sourceBgr, ink);
        //invert second plate when it is after Black
        if (inks.size() == 2u && i == 1 && isBlackInkSwatch(inks[0])) {
            cv::bitwise_not(pp.continuous, pp.continuous);
        }
        pp.tinted = tintPlateWithInkColor(pp.continuous, ink.bgr);
        pairs.push_back(std::move(pp));
    }
    // If no black is selected, solve per-pixel best ink mixing across all selected inks so colors not directly represented by one swatch can still be approximated by combinations.
    if (!containsBlackInk(inks)) {
        solveNoBlackBestMixPlates(pairs, sourceBgr, inks);
        for (size_t i = 0; i < pairs.size(); ++i) {
            pairs[i].tinted = tintPlateWithInkColor(pairs[i].continuous, inks[i].bgr);
        }
    } else if (inks.size() > 1u) {
        const int blackIdx = findBlackInkIndex(inks);
        if (blackIdx >= 0) {
            applyChromaAwareBlackSuppression(pairs, sourceBgr, blackIdx);
            for (size_t i = 0; i < pairs.size(); ++i) {
                pairs[i].tinted = tintPlateWithInkColor(pairs[i].continuous, inks[i].bgr);
            }
        }
    }
    return pairs;
}

cv::Mat buildCompositeFromPlates(const std::vector<PlatePair>& plates) {
    std::vector<cv::Mat> tintedLayers;
    tintedLayers.reserve(plates.size());
    for (const PlatePair& pp : plates) {
        tintedLayers.push_back(pp.tinted);
    }
    return multiplyBlendLayers(tintedLayers);
}

void applyDepthMonoMaskToPlates(
    std::vector<PlatePair>& plates,
    const cv::Mat& depthU8,
    int nearThreshold,
    bool multiInkIsNearRegion,
    int monoInkIndexInList,
    const std::vector<RisoInk>& inks) {
    if (plates.empty() || depthU8.empty() || plates.size() != inks.size()) {
        return;
    }
    CV_Assert(depthU8.type() == CV_8UC1);
    CV_Assert(depthU8.rows == plates[0].continuous.rows && depthU8.cols == plates[0].continuous.cols);
    if (monoInkIndexInList < 0 || monoInkIndexInList >= static_cast<int>(plates.size())) {
        return;
    }

    for (int y = 0; y < depthU8.rows; ++y) {
        for (int x = 0; x < depthU8.cols; ++x) {
            const bool isNear = static_cast<int>(depthU8.at<uint8_t>(y, x)) > nearThreshold;
            const bool useMultiInks = multiInkIsNearRegion ? isNear : !isNear;
            if (useMultiInks) {
                continue;
            }
            for (int i = 0; i < static_cast<int>(plates.size()); ++i) {
                if (i != monoInkIndexInList) {
                    plates[static_cast<size_t>(i)].continuous.at<uint8_t>(y, x) = 0;
                }
            }
        }
    }

    for (size_t i = 0; i < plates.size(); ++i) {
        plates[i].tinted = tintPlateWithInkColor(plates[i].continuous, inks[i].bgr);
    }
}

void applyDepthGradientMaskToPlates(
    std::vector<PlatePair>& plates,
    const cv::Mat& depthU8,
    int monoInkIndexInList,
    const std::vector<RisoInk>& inks) {
    if (plates.empty() || depthU8.empty() || plates.size() != inks.size()) {
        return;
    }
    CV_Assert(depthU8.type() == CV_8UC1);
    CV_Assert(depthU8.rows == plates[0].continuous.rows && depthU8.cols == plates[0].continuous.cols);
    if (monoInkIndexInList < 0 || monoInkIndexInList >= static_cast<int>(plates.size())) {
        monoInkIndexInList = 0;
    }

    const int n = static_cast<int>(plates.size());
    std::vector<std::pair<uint8_t, int>> strengths(static_cast<size_t>(n));
    std::vector<uint8_t> keep(static_cast<size_t>(n), 0);

    for (int y = 0; y < depthU8.rows; ++y) {
        for (int x = 0; x < depthU8.cols; ++x) {
            const float near01 = static_cast<float>(depthU8.at<uint8_t>(y, x)) / 255.f;
            const int kActive = std::clamp(1 + static_cast<int>(std::round(near01 * static_cast<float>(n - 1))), 1, n);

            std::fill(keep.begin(), keep.end(), static_cast<uint8_t>(0));
            keep[static_cast<size_t>(monoInkIndexInList)] = 1;

            if (kActive > 1) {
                for (int i = 0; i < n; ++i) {
                    strengths[static_cast<size_t>(i)] = std::make_pair(
                        plates[static_cast<size_t>(i)].continuous.at<uint8_t>(y, x), i);
                }
                std::sort(strengths.begin(), strengths.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

                int selected = 1;
                for (const auto& s : strengths) {
                    if (selected >= kActive) {
                        break;
                    }
                    if (s.second == monoInkIndexInList) {
                        continue;
                    }
                    keep[static_cast<size_t>(s.second)] = 1;
                    ++selected;
                }
            }

            for (int i = 0; i < n; ++i) {
                if (!keep[static_cast<size_t>(i)]) {
                    plates[static_cast<size_t>(i)].continuous.at<uint8_t>(y, x) = 0;
                }
            }
        }
    }

    for (size_t i = 0; i < plates.size(); ++i) {
        plates[i].tinted = tintPlateWithInkColor(plates[i].continuous, inks[i].bgr);
    }
}

cv::Mat buildMontage(
    const cv::Mat& sourceBgr,
    const std::vector<PlatePair>& plates,
    const std::vector<RisoInk>& inks,
    const cv::Mat& composite,
    int cellPx) {
    const int labelH = (cellPx > 0) ? 40 : std::clamp(sourceBgr.cols / 22, 36, 120);
    const int sepW = 10;
    const cv::Scalar sepColor(210, 210, 210);

    auto prepare = [cellPx](const cv::Mat& img) -> cv::Mat {
        cv::Mat bgr = (img.channels() == 1)
            ? [&] {
                  cv::Mat b;
                  cv::cvtColor(img, b, cv::COLOR_GRAY2BGR);
                  return b;
              }()
            : img.clone();
        if (cellPx <= 0) {
            return bgr;
        }
        cv::Mat out;
        const int interp = (cellPx < std::min(bgr.rows, bgr.cols)) ? cv::INTER_AREA : cv::INTER_LINEAR;
        cv::resize(bgr, out, cv::Size(cellPx, cellPx), 0.0, 0.0, interp);
        return out;
    };

    std::vector<cv::Mat> segments;
    segments.push_back(addCaptionStripBelow(prepare(sourceBgr), "Input (original)", labelH));

    for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
        const std::string cap = "Plate: " + inks[static_cast<size_t>(i)].name + " (continuous-tone)";
        segments.push_back(
            addCaptionStripBelow(prepare(plates[static_cast<size_t>(i)].continuous), cap, labelH));
    }

    segments.push_back(addCaptionStripBelow(prepare(composite), "Composite (multiply overprint)", labelH));

    int maxH = 0;
    for (const cv::Mat& s : segments) {
        maxH = std::max(maxH, s.rows);
    }
    for (cv::Mat& s : segments) {
        s = padPanelToHeight(s, maxH);
    }

    std::vector<cv::Mat> withSeps;
    withSeps.reserve(segments.size() * 2);
    for (int i = 0; i < static_cast<int>(segments.size()); ++i) {
        withSeps.push_back(segments[static_cast<size_t>(i)]);
        if (i + 1 < static_cast<int>(segments.size())) {
            withSeps.push_back(cv::Mat(maxH, sepW, CV_8UC3, sepColor));
        }
    }

    cv::Mat out;
    cv::hconcat(withSeps, out);
    return out;
}

std::string safeSpotFileStem(const std::string& inkName) {
    std::string s;
    for (char c : inkName) {
        s += (c == ' ' || c == '/') ? '_' : c;
    }
    return s;
}

}  // namespace

// ── Public API ───────────────────────────────────────────────────────────────

cv::Mat buildDitheredRisoComposite(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks) {
    // NOTE: despite the legacy name "dithered", this now returns a continuous-
    // tone composite.  Rename in a future refactor if desired.
    if (sourceBgr.empty() || inks.empty()) {
        return {};
    }
    const auto plates = buildAllPlates(sourceBgr, inks);
    return buildCompositeFromPlates(plates);
}

cv::Mat buildDitheredRisoCompositeDepthSplit(
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks,
    const cv::Mat& depthU8,
    int nearThreshold,
    bool multiInkIsNearRegion,
    int monoInkIndexInList) {
    if (sourceBgr.empty() || inks.empty() || depthU8.empty()) {
        return {};
    }
    if (depthU8.type() != CV_8UC1 || depthU8.rows != sourceBgr.rows || depthU8.cols != sourceBgr.cols) {
        return {};
    }
    auto plates = buildAllPlates(sourceBgr, inks);
    (void)nearThreshold;
    (void)multiInkIsNearRegion;
    applyDepthGradientMaskToPlates(plates, depthU8, monoInkIndexInList, inks);
    return buildCompositeFromPlates(plates);
}

cv::Mat buildDitheredRisoCompositeFaceSplit(const cv::Mat& sourceBgr, const std::vector<RisoInk>& inks) {
    if (sourceBgr.empty() || inks.empty()) {
        return {};
    }
    auto plates = buildAllPlates(sourceBgr, inks);
    if (plates.empty()) {
        return {};
    }

    cv::Mat fullComposite = buildCompositeFromPlates(plates);
    if (fullComposite.empty()) {
        return {};
    }

    cv::Mat gray;
    cv::cvtColor(sourceBgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    detectFaces(gray, faces);
    const cv::Rect faceSquare = largestFaceSquareRoi(sourceBgr.size(), faces);
    if (faceSquare.area() <= 0) {
        return fullComposite;
    }

    const int bgIdx = leastPresentInkIndex(plates);
    if (bgIdx < 0) {
        return fullComposite;
    }
    // Background is a single flat swatch: the least-present selected ink color only.
    cv::Mat backgroundOnly(sourceBgr.rows, sourceBgr.cols, CV_8UC3, cv::Scalar(
        inks[static_cast<size_t>(bgIdx)].bgr[0],
        inks[static_cast<size_t>(bgIdx)].bgr[1],
        inks[static_cast<size_t>(bgIdx)].bgr[2]));

    cv::Mat out = backgroundOnly.clone();
    fullComposite(faceSquare).copyTo(out(faceSquare));
    return out;
}

bool savePrintTogetherPreview(
    const std::string& outputPath,
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks,
    const cv::Mat& compositeBgr,
    int cellPx) {
    if (sourceBgr.empty() || inks.empty() || compositeBgr.empty()) {
        return false;
    }
    const auto plates = buildAllPlates(sourceBgr, inks);
    const cv::Mat montage = buildMontage(sourceBgr, plates, inks, compositeBgr, cellPx);
    if (montage.empty()) {
        return false;
    }
    return image_utils::saveImage(outputPath, montage);
}

SeparationExportResult exportSeparationPackage(
    const std::string& separationDir,
    const cv::Mat& sourceBgr,
    const std::vector<RisoInk>& inks) {
    SeparationExportResult result;
    if (sourceBgr.empty() || inks.empty()) {
        return result;
    }
    if (!image_utils::ensureDirectory(separationDir)) {
        return result;
    }

    const auto plates = buildAllPlates(sourceBgr, inks);
    result.composite = buildCompositeFromPlates(plates);

    if (!image_utils::saveImage(separationDir + "/preview_softproof.png", result.composite)) {
        return result;
    }

    {
        std::ofstream ord(separationDir + "/PRINT_ORDER.txt");
        if (!ord) {
            return result;
        }
        ord << "# Spot channel grayscale PNGs (continuous-tone masters).\n"
            << "# Each plate: 0 = white (no ink), 255 = full ink coverage.\n"
            << "# Dither / screen these files before sending to the Riso RZ/EZ.\n"
            << "# https://colorshift.theretherenow.com/how-to-use\n\n";
        for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
            ord << "Spot_" << std::setw(2) << std::setfill('0') << (i + 1) << ": "
                << inks[static_cast<size_t>(i)].name << '\n';
        }
    }

    result.spotPaths.reserve(inks.size());
    for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
        std::ostringstream fn;
        fn << separationDir << "/Spot_" << std::setw(2) << std::setfill('0') << (i + 1) << '_'
           << safeSpotFileStem(inks[static_cast<size_t>(i)].name) << ".png";
        const std::string path = fn.str();
        if (image_utils::saveImage(path, plates[static_cast<size_t>(i)].continuous)) {
            result.spotPaths.push_back(path);
        }
    }

    return result;
}

std::string makeInkComboSlug(const std::vector<RisoInk>& inks) {
    std::ostringstream o;
    for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
        if (i > 0) {
            o << '-';
        }
        for (char c : inks[static_cast<size_t>(i)].name) {
            o << ((c == ' ' || c == '/') ? '_' : c);
        }
    }
    return o.str();
}

void computeCmykPlanes(const cv::Mat& bgr, cv::Mat& outC, cv::Mat& outM, cv::Mat& outY, cv::Mat& outK) {
    outC.create(bgr.size(), CV_8UC1);
    outM.create(bgr.size(), CV_8UC1);
    outY.create(bgr.size(), CV_8UC1);
    outK.create(bgr.size(), CV_8UC1);

    for (int y = 0; y < bgr.rows; ++y) {
        for (int x = 0; x < bgr.cols; ++x) {
            const cv::Vec3b px = bgr.at<cv::Vec3b>(y, x);
            const double b = static_cast<double>(px[0]) / 255.0;
            const double g = static_cast<double>(px[1]) / 255.0;
            const double r = static_cast<double>(px[2]) / 255.0;
            const double k = 1.0 - std::max({r, g, b});

            if (k >= 1.0 - 1e-6) {
                outC.at<uint8_t>(y, x) = 0;
                outM.at<uint8_t>(y, x) = 0;
                outY.at<uint8_t>(y, x) = 0;
                outK.at<uint8_t>(y, x) = 255;
                continue;
            }
            const double inv = 1.0 / (1.0 - k);
            outC.at<uint8_t>(y, x) =
                static_cast<uint8_t>(std::lround(std::clamp((1.0 - r - k) * inv, 0.0, 1.0) * 255.0));
            outM.at<uint8_t>(y, x) =
                static_cast<uint8_t>(std::lround(std::clamp((1.0 - g - k) * inv, 0.0, 1.0) * 255.0));
            outY.at<uint8_t>(y, x) =
                static_cast<uint8_t>(std::lround(std::clamp((1.0 - b - k) * inv, 0.0, 1.0) * 255.0));
            outK.at<uint8_t>(y, x) = static_cast<uint8_t>(std::lround(k * 255.0));
        }
    }
}

cv::Mat extractCMYKChannel(const cv::Mat& bgr, int channelIndex) {
    cv::Mat C, M, Y, K;
    computeCmykPlanes(bgr, C, M, Y, K);
    if (channelIndex == 0) {
        return C;
    }
    if (channelIndex == 1) {
        return M;
    }
    if (channelIndex == 2) {
        return Y;
    }
    return K;
}

cv::Mat extractRGBChannel(const cv::Mat& bgr, int channelIndex) {
    cv::Mat ch;
    cv::extractChannel(bgr, ch, channelIndex);
    return ch;
}

// ── Atkinson dither (kept for print-master output) ───────────────────────────
// This is retained so callers can optionally dither a continuous-tone plate
// before sending it to the Riso printer.  It is no longer called internally
// during composite generation.
cv::Mat atkinsonDither(const cv::Mat& grayU8) {
    CV_Assert(grayU8.type() == CV_8UC1);
    cv::Mat work;
    grayU8.convertTo(work, CV_32F);

    auto distribute = [](cv::Mat& img, int x, int y, float amount) {
        if (amount == 0.f) {
            return;
        }
        if (x >= 0 && x < img.cols && y >= 0 && y < img.rows) {
            img.at<float>(y, x) += amount;
        }
    };

    for (int y = 0; y < work.rows; ++y) {
        for (int x = 0; x < work.cols; ++x) {
            float& old = work.at<float>(y, x);
            const uint8_t newv = old < 128.f ? 0 : 255;
            const float err = old - static_cast<float>(newv);
            old = static_cast<float>(newv);

            const float e = err / 8.f;
            distribute(work, x + 1, y, e);
            distribute(work, x + 2, y, e);
            distribute(work, x - 1, y + 1, e);
            distribute(work, x, y + 1, e);
            distribute(work, x + 1, y + 1, e);
            distribute(work, x, y + 2, e);
        }
    }
    cv::Mat out;
    work.convertTo(out, CV_8UC1);
    return out;
}

}
