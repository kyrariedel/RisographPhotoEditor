/*
Kyra Riedel
April 23, 2026
Interactive image-edit and preview pipeline.
*/

#include "img_pipeline.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "faceDetect.h"
#include "filter.h"
#include "riso_inks.h"
#include "riso_output.h"

namespace {

std::string fileBasename(const std::string& path) {
    if (path.empty()) {
        return "Image";
    }
    const auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

std::vector<RisoInk> parseInkIndicesLine(const std::string& line, const std::vector<RisoInk>& palette) {
    std::vector<RisoInk> inks;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t')) {
            tok.erase(0, 1);
        }
        while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t')) {
            tok.pop_back();
        }
        if (tok.empty()) {
            continue;
        }
        try {
            const int idx = std::stoi(tok);
            if (idx >= 0 && idx < static_cast<int>(palette.size())) {
                inks.push_back(palette[static_cast<size_t>(idx)]);
            }
        } catch (...) {
        }
    }
    return inks;
}

}  // namespace

cv::Mat runInteractiveImagePipeline(const cv::Mat& imageBgr, const std::string& sourceImagePath) {
    if (imageBgr.empty() || imageBgr.channels() != 3) {
        return {};
    }

    cv::Mat original = imageBgr.clone();
    cv::Mat current = original.clone();
    cv::Mat sobelOutput;
    cv::Mat blurQuantImg;
    cv::Mat depthRawCache;

    const std::string windowName = fileBasename(sourceImagePath);
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::imshow(windowName, current);

    char filter = 'o';

    float brightness = 0.0f;
    float contrast = 0.0f;

    int edgeMagThr = 55;
    int edgeColorIdx = 0;

    std::cout << "\n--- Image pipeline (then RISO on result after you press 'q') ---\n";
    std::cout << "Filters: 'o' original, 'g' opencv gray, 'h' custom gray, 'a' sepia, 'x' sobel X, 'y' sobel Y, "
                 "'m' magnitude, 'e' emboss, 'd' depth viz, '2' fog, 'c' colorful faces, 'l' blur quantize, "
                 "'b' blur 5x5 (sep), 'n' blur 5x5 (int), 't' sobel edges (black), 'u' sobel edges (color)\n";
    std::cout << "Depth: 'w' refresh raw depth. RISO preview on current image: 'p' full inks, 'v' depth-split, "
                 "'f' face-split.\n";
    std::cout << "'+'/'-' brightness, '['/']' contrast, 'r' reset BC, ','/'.' edge threshold, 'k' cycle edge color\n";
    std::cout << "'s' save JPG, 'q' quit and use this frame for RISO scoring/export\n\n";

    while (true) {
        const int keyCode = cv::waitKey(30);
        const char key = static_cast<char>(keyCode);

        if (filter == 'g') {
            openCVGray(original, current);
        } else if (filter == 'h') {
            customGray(original, current);
        } else if (filter == 'a') {
            sepia(original, current);
        } else if (filter == 'x') {
            sobelX3x3(original, sobelOutput);
            cv::convertScaleAbs(sobelOutput, current);
            cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);
        } else if (filter == 'y') {
            sobelY3x3(original, sobelOutput);
            cv::convertScaleAbs(sobelOutput, current);
            cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);
        } else if (filter == 'd') {
            depthEstimation(original, current);
            if (computeDepthMapU8(original, depthRawCache) != 0) {
                depthRawCache.release();
            }
        } else if (filter == 'b') {
            blur5x5_2(original, current);
        } else if (filter == 'n') {
            blur5x5_1(original, current);
        } else if (filter == '2') {
            addFogDepth(original, current);
            if (computeDepthMapU8(original, depthRawCache) != 0) {
                depthRawCache.release();
            }
        } else if (filter == 'm') {
            cv::Mat sobelX16;
            cv::Mat sobelY16;
            sobelX3x3(original, sobelX16);
            sobelY3x3(original, sobelY16);
            gradientMagnitude(sobelX16, sobelY16, current);
        } else if (filter == 'c') {
            colorfulFaces(original, current);
        } else if (filter == 'e') {
            emboss(original, current);
            cv::convertScaleAbs(current, current);
            cv::cvtColor(current, current, cv::COLOR_BGR2GRAY);
        } else if (filter == 'l') {
            blurQuantize(original, blurQuantImg, 10);
            current = blurQuantImg;
        } else if (filter == 't') {
            sobelEdgeHighlight(original, current, edgeMagThr, cv::Vec3b(0, 0, 0), true);
        } else if (filter == 'u') {
            const std::vector<RisoInk> pal = getRisoPalette();
            const cv::Vec3b edgeBgr = pal[static_cast<size_t>(edgeColorIdx % static_cast<int>(pal.size()))].bgr;
            sobelEdgeHighlight(original, current, edgeMagThr, edgeBgr, false);
        } else if (filter == 'o') {
            current = original.clone();
        }

        if (brightness != 0.0f || contrast != 0.0f) {
            cv::Mat adjusted;
            adjustBrightnessContrast(current, adjusted, brightness, contrast);
            current = adjusted;
        }

        cv::imshow(windowName, current);

        if (key == 'q') {
            break;
        }
        if (key == 'g') {
            filter = 'g';
        }
        if (key == 'b') {
            filter = 'b';
        }
        if (key == 'n') {
            filter = 'n';
        }
        if (key == 'h') {
            filter = 'h';
        }
        if (key == 'x') {
            filter = 'x';
        }
        if (key == 'y') {
            filter = 'y';
        }
        if (key == 'a') {
            filter = 'a';
        }
        if (key == 'd') {
            filter = 'd';
        }
        if (key == 'o') {
            filter = 'o';
        }
        if (key == '2') {
            filter = '2';
        }
        if (key == 'c') {
            filter = 'c';
        }
        if (key == 'e') {
            filter = 'e';
        }
        if (key == 'm') {
            filter = 'm';
        }
        if (key == 'l') {
            filter = 'l';
        }
        if (key == 't') {
            filter = 't';
        }
        if (key == 'u') {
            filter = 'u';
        }

        if (key == 'w') {
            if (computeDepthMapU8(original, depthRawCache) == 0) {
                std::cout << "Depth cache updated (" << depthRawCache.cols << "x" << depthRawCache.rows << ").\n";
            } else {
                std::cerr << "Depth cache failed (set RISO_DA2_MODEL to your model_fp16.onnx path).\n";
            }
        }

        if (key == 'k') {
            edgeColorIdx = (edgeColorIdx + 1) % 6;
            std::cout << "Edge highlight color index (RISO palette): " << edgeColorIdx << "\n";
        }

        if (key == ',' || key == '<') {
            edgeMagThr = std::max(5, edgeMagThr - 5);
            std::cout << "Edge magnitude threshold (0-255 scaled): " << edgeMagThr << "\n";
        } else if (key == '.' || key == '>') {
            edgeMagThr = std::min(250, edgeMagThr + 5);
            std::cout << "Edge magnitude threshold (0-255 scaled): " << edgeMagThr << "\n";
        }

        if (key == 'p') {
            const std::vector<RisoInk> palette = getRisoPalette();
            std::cout << "RISO palette indices: 0 Red, 1 Yellow, 2 Pink, 3 Blue, 4 Black, 5 Green\n";
            std::cout << "Enter ink indices in print order (CSV, e.g. 3,2): " << std::flush;
            std::string line;
            std::getline(std::cin >> std::ws, line);
            const std::vector<RisoInk> inks = parseInkIndicesLine(line, palette);
            if (inks.empty()) {
                std::cout << "No valid inks; skipping RISO preview.\n";
            } else {
                const cv::Mat riso = riso_output::buildDitheredRisoComposite(current, inks);
                if (!riso.empty()) {
                    cv::namedWindow("Riso preview", cv::WINDOW_AUTOSIZE);
                    cv::imshow("Riso preview", riso);
                    current = riso.clone();
                    std::cout << "RISO preview applied to working image.\n";
                }
            }
        }

        if (key == 'v') {
            const std::vector<RisoInk> palette = getRisoPalette();
            if (depthRawCache.empty() || depthRawCache.size() != original.size()) {
                if (computeDepthMapU8(original, depthRawCache) != 0) {
                    std::cerr << "Depth-split needs depth map ('w' or 'd'/'2'), or set RISO_DA2_MODEL.\n";
                }
            }
            if (depthRawCache.empty()) {
                std::cout << "Skipping depth-split RISO preview.\n";
            } else {
                std::cout << "Enter ink indices in print order (CSV): " << std::flush;
                std::string line;
                std::getline(std::cin >> std::ws, line);
                const std::vector<RisoInk> inks = parseInkIndicesLine(line, palette);
                if (inks.size() < 2) {
                    std::cout << "Need at least two inks. Skipping.\n";
                } else {
                    std::cout << "Mono-ink index in your list (0.." << (inks.size() - 1)
                              << ", used in farthest/single-ink region): " << std::flush;
                    int monoIdx = 0;
                    std::cin >> monoIdx;

                    if (monoIdx < 0 || monoIdx >= static_cast<int>(inks.size())) {
                        monoIdx = 0;
                    }

                    const cv::Mat riso = riso_output::buildDitheredRisoCompositeDepthSplit(
                        current, inks, depthRawCache, 128, true, monoIdx);
                    if (!riso.empty()) {
                        cv::namedWindow("Riso preview", cv::WINDOW_AUTOSIZE);
                        cv::imshow("Riso preview", riso);
                        current = riso.clone();
                        std::cout << "Depth-gradient RISO preview applied (near=most inks, far=mono).\n";
                    }
                }
            }
        }

        if (key == 'f') {
            const std::vector<RisoInk> palette = getRisoPalette();
            std::cout << "Enter ink indices in print order (CSV): " << std::flush;
            std::string line;
            std::getline(std::cin >> std::ws, line);
            const std::vector<RisoInk> inks = parseInkIndicesLine(line, palette);
            if (inks.empty()) {
                std::cout << "No valid inks; skipping face-split RISO preview.\n";
            } else {
                const cv::Mat riso = riso_output::buildDitheredRisoCompositeFaceSplit(current, inks);
                if (!riso.empty()) {
                    cv::namedWindow("Riso preview", cv::WINDOW_AUTOSIZE);
                    cv::imshow("Riso preview", riso);
                    current = riso.clone();
                    std::cout << "Face-split RISO preview applied.\n";
                } else {
                    std::cout << "Face-split RISO preview failed.\n";
                }
            }
        }

        if (key == 's') {
            std::string name;
            std::cout << "Enter filename (no extension): ";
            std::cin >> name;
            const std::string filename = name + ".jpg";
            if (cv::imwrite(filename, current)) {
                std::cout << "Saved: " << filename << "\n";
            } else {
                std::cout << "Save failed.\n";
            }
        }

        if (key == '+' || key == '=') {
            brightness = std::min(100.0f, brightness + 5.0f);
        } else if (key == '-' || key == '_') {
            brightness = std::max(-100.0f, brightness - 5.0f);
        } else if (key == '[' || key == '{') {
            contrast = std::min(100.0f, contrast + 5.0f);
        } else if (key == ']' || key == '}') {
            contrast = std::max(-100.0f, contrast - 5.0f);
        } else if (key == 'r') {
            brightness = contrast = 0.0f;
        }
    }

    cv::Mat out = current.clone();
    if (out.channels() == 1) {
        cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
    }
    cv::destroyAllWindows();
    return out;
}
