/*
Kyra Riedel
April 23, 2026
CLI entry point for scoring and export.
*/

#include "image_utils.h"
#include "img_pipeline.h"
#include "riso_inks.h"
#include "riso_output.h"
#include "scoring.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void printUsage(const char* programName) {
    std::cout << "Usage:\n"
              << "  " << programName << " --image <path> --layers <1-6> [--inks <indices|all>] [--edit]\n"
              << "  --edit   OpenCV UI: Project 1 image filters + optional RISO previews; result is scored/exported.\n"
              << "  --inks is optional; omit it (or use 'all') to search all 6 palette inks.\n"
              << "Depth model: env RISO_DA2_MODEL, else search ./ ../ ../../ for model_fp16.onnx. "
                 "Faces: RISO_FACE_CASCADE, else ./ or ../haarcascade_frontalface_alt2.xml.\n"
              << "Example:\n"
              << "  " << programName << " --image input.jpg --layers 3\n"
              << "  " << programName << " --image input.jpg --layers 3 --inks 0,3,5 --edit\n"
              << "If no args are provided, interactive prompt mode is used.\n\n";
}

std::vector<int> parseInkIndices(const std::string& csv) {
    std::vector<int> indices;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            indices.push_back(std::stoi(token));
        }
    }
    return indices;
}

void printPalette(const std::vector<RisoInk>& palette) {
    std::cout << "RISO Ink Palette\n";
    std::cout << "================\n";
    for (int i = 0; i < static_cast<int>(palette.size()); ++i) {
        const auto& ink = palette[i];
        const int r = ink.bgr[2];
        const int g = ink.bgr[1];
        const int b = ink.bgr[0];
        std::cout << std::setw(2) << i << ": " << ink.name << " (" << r << "," << g << "," << b << ")\n";
    }
    std::cout << '\n';
}

std::vector<RisoInk> pickSelectedInks(
    const std::vector<RisoInk>& palette,
    const std::vector<int>& selectedIndices) {
    std::vector<RisoInk> selected;
    for (int index : selectedIndices) {
        if (index < 0 || index >= static_cast<int>(palette.size())) {
            throw std::invalid_argument("Ink index out of range: " + std::to_string(index));
        }
        selected.push_back(palette[index]);
    }
    return selected;
}

std::vector<int> allPaletteIndices(int paletteSize) {
    std::vector<int> indices(static_cast<size_t>(paletteSize));
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

std::string joinInkNames(const std::vector<RisoInk>& inks) {
    std::ostringstream out;
    for (int i = 0; i < static_cast<int>(inks.size()); ++i) {
        out << inks[i].name;
        if (i + 1 < static_cast<int>(inks.size())) {
            out << ", ";
        }
    }
    return out.str();
}

std::string truncateForCaption(const std::string& s, std::size_t maxChars) {
    if (s.size() <= maxChars) {
        return s;
    }
    if (maxChars <= 3) {
        return s.substr(0, maxChars);
    }
    return s.substr(0, maxChars - 3) + "...";
}

void printSingleComboAccuracy(const ScoredCombination& only) {
    std::cout << "\nSpecified inks fix the layer set — exactly one combination (no search).\n";
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << only.scorePercent << "%\n";
    std::cout << "Inks (print order): " << joinInkNames(only.inks) << '\n';
}

void printMeanAccuracy(const std::vector<ScoredCombination>& scored) {
    if (scored.empty()) {
        return;
    }
    double sum = 0.0;
    for (const auto& s : scored) {
        sum += s.scorePercent;
    }
    const double mean = sum / static_cast<double>(scored.size());
    std::cout << "Mean accuracy across evaluated combinations: " << std::fixed << std::setprecision(2) << mean << "%\n";
}

void printTopBottomAccuracyTables(const std::vector<ScoredCombination>& scored) {
    const int n = static_cast<int>(scored.size());
    const int topN = std::min(3, n);
    const int botN = std::min(3, n);

    std::cout << "\n--- Top " << topN << " (highest accuracy) ---\n";
    std::cout << std::setw(5) << "Rank" << "  | " << std::setw(44) << std::left << "Inks (print order)"
              << std::right << " | " << std::setw(10) << "Accuracy %" << '\n';
    std::cout << std::string(72, '-') << '\n';
    for (int i = 0; i < topN; ++i) {
        std::string names = joinInkNames(scored[static_cast<size_t>(i)].inks);
        if (names.size() > 44) {
            names = names.substr(0, 41) + "...";
        }
        std::cout << std::setw(5) << (i + 1) << "  | " << std::setw(44) << std::left << names << std::right << " | "
                  << std::fixed << std::setprecision(2) << std::setw(10) << scored[static_cast<size_t>(i)].scorePercent
                  << '\n';
    }

    std::cout << "\n--- Bottom " << botN << " (lowest accuracy) ---\n";
    std::cout << std::setw(5) << "Rank" << "  | " << std::setw(44) << std::left << "Inks (print order)"
              << std::right << " | " << std::setw(10) << "Accuracy %" << '\n';
    std::cout << std::string(72, '-') << '\n';
    for (int b = 0; b < botN; ++b) {
        const int idx = n - 1 - b;
        const int rank = idx + 1;
        std::string names = joinInkNames(scored[static_cast<size_t>(idx)].inks);
        if (names.size() > 44) {
            names = names.substr(0, 41) + "...";
        }
        std::cout << std::setw(5) << rank << "  | " << std::setw(44) << std::left << names << std::right << " | "
                  << std::fixed << std::setprecision(2) << std::setw(10) << scored[static_cast<size_t>(idx)].scorePercent
                  << '\n';
    }
}

cv::Mat resizeToSquareCanvas(const cv::Mat& srcBgr, int side) {
    cv::Mat canvas(side, side, CV_8UC3, cv::Scalar(255, 255, 255));
    if (srcBgr.empty()) {
        return canvas;
    }
    cv::Mat bgr = (srcBgr.channels() == 3) ? srcBgr : [&] {
        cv::Mat t;
        cv::cvtColor(srcBgr, t, cv::COLOR_GRAY2BGR);
        return t;
    }();
    const double sx = static_cast<double>(side) / static_cast<double>(bgr.cols);
    const double sy = static_cast<double>(side) / static_cast<double>(bgr.rows);
    const double scale = std::min(sx, sy);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    const int ox = (side - resized.cols) / 2;
    const int oy = (side - resized.rows) / 2;
    resized.copyTo(canvas(cv::Rect(ox, oy, resized.cols, resized.rows)));
    return canvas;
}

cv::Mat addTwoLineCaptionBelow(const cv::Mat& contentBgr, const std::string& line1, const std::string& line2, int capH) {
    CV_Assert(contentBgr.type() == CV_8UC3);
    const int lh = std::max(40, capH);
    cv::Mat out(contentBgr.rows + lh, contentBgr.cols, CV_8UC3);
    contentBgr.copyTo(out(cv::Rect(0, 0, contentBgr.cols, contentBgr.rows)));
    out(cv::Rect(0, contentBgr.rows, contentBgr.cols, lh)).setTo(cv::Scalar(28, 28, 28));
    const double fs = std::clamp(static_cast<double>(contentBgr.cols) / 480.0, 0.38, 0.85);
    cv::putText(
        out,
        line1,
        cv::Point(8, contentBgr.rows + static_cast<int>(22 * fs) + 8),
        cv::FONT_HERSHEY_SIMPLEX,
        fs,
        cv::Scalar(250, 250, 250),
        1,
        cv::LINE_AA);
    cv::putText(
        out,
        line2,
        cv::Point(8, contentBgr.rows + lh - 8),
        cv::FONT_HERSHEY_SIMPLEX,
        fs * 0.92,
        cv::Scalar(220, 220, 220),
        1,
        cv::LINE_AA);
    return out;
}

cv::Mat blankPreviewCell(int squareSide, int capH, const std::string& label) {
    cv::Mat sq(squareSide, squareSide, CV_8UC3, cv::Scalar(230, 230, 230));
    return addTwoLineCaptionBelow(sq, label, "—", capH);
}

void openTopBottomCompositePreview(const std::vector<ScoredCombination>& scored, const std::string& imagePathForTitle) {
    const int n = static_cast<int>(scored.size());
    if (n < 2) {
        return;
    }

    constexpr int kCell = 280;
    constexpr int kCap = 56;
    constexpr std::size_t kMaxCaptionChars = 48;

    auto cellForTop = [&](int col) -> cv::Mat {
        if (col >= std::min(3, n)) {
            return blankPreviewCell(kCell, kCap, "—");
        }
        const ScoredCombination& s = scored[static_cast<size_t>(col)];
        cv::Mat sq = resizeToSquareCanvas(s.composite, kCell);
        std::ostringstream acc;
        acc << std::fixed << std::setprecision(1) << s.scorePercent << "%";
        const std::string line1 = truncateForCaption(joinInkNames(s.inks), kMaxCaptionChars);
        return addTwoLineCaptionBelow(sq, line1, acc.str(), kCap);
    };

    auto cellForBottom = [&](int col) -> cv::Mat {
        if (col >= std::min(3, n)) {
            return blankPreviewCell(kCell, kCap, "—");
        }
        const int idx = n - 1 - col;
        const ScoredCombination& s = scored[static_cast<size_t>(idx)];
        cv::Mat sq = resizeToSquareCanvas(s.composite, kCell);
        std::ostringstream acc;
        acc << std::fixed << std::setprecision(1) << s.scorePercent << "%";
        const std::string line1 = truncateForCaption(joinInkNames(s.inks), kMaxCaptionChars);
        return addTwoLineCaptionBelow(sq, line1, acc.str(), kCap);
    };

    cv::Mat r0;
    cv::hconcat(std::vector<cv::Mat>{cellForTop(0), cellForTop(1), cellForTop(2)}, r0);
    cv::Mat r1;
    cv::hconcat(std::vector<cv::Mat>{cellForBottom(0), cellForBottom(1), cellForBottom(2)}, r1);

    cv::Mat grid;
    cv::vconcat(r0, r1, grid);

    std::string base = std::filesystem::path(imagePathForTitle).filename().string();
    if (base.empty()) {
        base = "image";
    }
    const std::string win = base + ": top3 vs bottom 3";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
    cv::imshow(win, grid);
    std::cout << "\nOpened preview window.\n";
    std::cout << "Click the preview window and press any key to close and continue...\n";
    cv::waitKey(0);
    cv::destroyWindow(win);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::vector<RisoInk> palette = getRisoPalette();

        std::string imagePath;
        int layerCount = 0;
        std::vector<int> selectedIndices;
        bool useImageEditPipeline = false;

        if (argc > 1) {
            for (int i = 1; i < argc; ++i) {
                const std::string arg = argv[i];
                if (arg == "--image" && i + 1 < argc) {
                    imagePath = argv[++i];
                } else if (arg == "--layers" && i + 1 < argc) {
                    layerCount = std::stoi(argv[++i]);
                } else if (arg == "--edit") {
                    useImageEditPipeline = true;
                } else if (arg == "--inks") {
                    if (i + 1 < argc) {
                        const std::string v = argv[++i];
                        if (v == "all" || v.empty()) {
                            selectedIndices.clear();
                        } else {
                            selectedIndices = parseInkIndices(v);
                        }
                    }
                } else if (arg == "--help" || arg == "-h") {
                    printUsage(argv[0]);
                    return 0;
                } else {
                    std::cerr << "Unknown or incomplete argument: " << arg << "\n\n";
                    printUsage(argv[0]);
                    return 1;
                }
            }
        } else {
            printPalette(palette);
            std::cout << "Input image path: ";
            std::getline(std::cin, imagePath);

            std::cout << "Number of ink layers (1-6): ";
            std::string layersInput;
            std::getline(std::cin, layersInput);
            layerCount = std::stoi(layersInput);

            std::cout << "Select ink indices 0-5 (comma-separated), or press Enter for all 6: ";
            std::string indicesInput;
            std::getline(std::cin, indicesInput);
            if (!indicesInput.empty()) {
                selectedIndices = parseInkIndices(indicesInput);
            }
        }

        if (imagePath.empty()) {
            throw std::invalid_argument("Image path is required.");
        }
        if (layerCount < 1 || layerCount > 6) {
            throw std::invalid_argument("Layer count must be between 1 and 6.");
        }

        if (selectedIndices.empty()) {
            selectedIndices = allPaletteIndices(static_cast<int>(palette.size()));
        }

        const std::vector<RisoInk> selectedInks = pickSelectedInks(palette, selectedIndices);
        if (layerCount > static_cast<int>(selectedInks.size())) {
            throw std::invalid_argument("Layer count cannot exceed number of selected inks.");
        }

        cv::Mat sourceFull = image_utils::loadImageBgr(imagePath);
        if (useImageEditPipeline) {
            std::cout << "Opening image editor; press 'q' when done to run RISO on the edited image.\n";
            sourceFull = runInteractiveImagePipeline(sourceFull, imagePath);
            if (sourceFull.empty()) {
                throw std::runtime_error("Image pipeline returned an empty image.");
            }
        }
        const std::size_t totalCombinations =
            countInkCombinations(selectedInks.size(), static_cast<std::size_t>(layerCount));
        const bool singleForcedCombo = (totalCombinations == 1);

        if (!singleForcedCombo && totalCombinations > 200u) {
            std::cout << "\nNote: " << totalCombinations
                      << " ink combinations exist; scoring a random sample of 200.\n";
            std::cout << "      Top / bottom summaries reflect that sample only.\n";
        }

        std::vector<ScoredCombination> scored =
            evaluateInkCombinations(sourceFull, selectedInks, layerCount, 200);
        if (scored.empty()) {
            throw std::runtime_error("No ink combinations could be scored.");
        }

        const std::string outputDir = "./riso_output";
        if (!image_utils::ensureDirectory(outputDir)) {
            return 1;
        }

        std::vector<std::string> spotPaths;
        const std::string comboSlug = riso_output::makeInkComboSlug(scored[0].inks);
        const std::string sepDir = outputDir + "/separation_" + comboSlug;
        riso_output::SeparationExportResult exported =
            riso_output::exportSeparationPackage(sepDir, sourceFull, scored[0].inks);
        spotPaths = std::move(exported.spotPaths);
        if (!exported.composite.empty()) {
            riso_output::savePrintTogetherPreview(sepDir + "/layout_proof.png", sourceFull, scored[0].inks, exported.composite, 0);
        }

        if (singleForcedCombo) {
            printSingleComboAccuracy(scored[0]);
            printMeanAccuracy(scored);
        } else {
            printTopBottomAccuracyTables(scored);
            printMeanAccuracy(scored);
            openTopBottomCompositePreview(scored, imagePath);
        }

        std::cout << "\nSeparation files on disk are for rank 1 (best) only: separation_" << comboSlug << "/\n";
        std::cout << "  PRINT_ORDER.txt, Spot_##_<Ink>.png, preview_softproof.png, layout_proof.png\n";
        for (int i = 0; i < static_cast<int>(spotPaths.size()); ++i) {
            std::cout << "  " << spotPaths[i] << '\n';
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
