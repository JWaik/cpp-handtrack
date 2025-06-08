#include <opencv2/opencv.hpp>
#include "SkinSegmenter.hpp"
#include "ContourAnalyzer.hpp"
#include "MediaPipeDetector.hpp"
#include "MLClassifier.hpp"
#include "Utils.hpp"
#include "cxxopts.hpp"

// HSV thresholds
int minH = 0, minS = 30, minV = 60;
int maxH = 20, maxS = 150, maxV = 255;

// YCrCb thresholds
int minY = 0, minCr = 133, minCb = 77;
int maxY = 255, maxCr = 173, maxCb = 127;

// Finger-cout tuning
int angleThreshold = 80;
int depthThreshold = 20;
int startRatioSlider = 30;  // Represents 0.30
int farRatioSlider = 20;    // Represents 0.20

void setupTrackbarsHSV() {
    cv::namedWindow("ControlsHSV", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Min H", "ControlsHSV", &minH, 179);
    cv::createTrackbar("Max H", "ControlsHSV", &maxH, 179);
    cv::createTrackbar("Min S", "ControlsHSV", &minS, 255);
    cv::createTrackbar("Max S", "ControlsHSV", &maxS, 255);
    cv::createTrackbar("Min V", "ControlsHSV", &minV, 255);
    cv::createTrackbar("Max V", "ControlsHSV", &maxV, 255);
    cv::createTrackbar("Angle <", "ControlsHSV", &angleThreshold, 180);
    cv::createTrackbar("Depth >", "ControlsHSV", &depthThreshold, 20);
    cv::createTrackbar("Start Ratio x100", "ControlsHSV", &startRatioSlider, 100);
    cv::createTrackbar("Far Ratio x100", "ControlsHSV", &farRatioSlider, 100);
}

void setupTrackbarsYCrCb() {
    cv::namedWindow("ControlsYCrCb", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Min Y", "ControlsYCrCb", &minY, 255);
    cv::createTrackbar("Max Y", "ControlsYCrCb", &maxY, 255);
    cv::createTrackbar("Min Cr", "ControlsYCrCb", &minCr, 255);
    cv::createTrackbar("Max Cr", "ControlsYCrCb", &maxCr, 255);
    cv::createTrackbar("Min Cb", "ControlsYCrCb", &minCb, 255);
    cv::createTrackbar("Max Cb", "ControlsYCrCb", &maxCb, 255);
    cv::createTrackbar("Angle <", "ControlsYCrCb", &angleThreshold, 180);
    cv::createTrackbar("Depth >", "ControlsYCrCb", &depthThreshold, 20);
    cv::createTrackbar("Start Ratio x100", "ControlsYCrCb", &startRatioSlider, 100);
    cv::createTrackbar("Far Ratio x100", "ControlsYCrCb", &farRatioSlider, 100);
}

void initSkinDetect(ColorSpace colorSpace) {
    if (colorSpace == ColorSpace::YCrCb) {
        setupTrackbarsYCrCb();
    } else {
        setupTrackbarsHSV();
    }
}

void runSkinDetect(cv::Mat &frame, SkinSegmenter &segSkin, ColorSpace colorSpace, cv::Mat &output) {
    ThresholdRange tr;
    if (colorSpace == ColorSpace::HSV) {
        tr = {minH, minS, minV, maxH, maxS, maxV};
    } else {
        tr = {minY, minCr, minCb, maxY, maxCr, maxCb};
    }

    segSkin.setThresholds(tr);
    segSkin.segment(frame, output);
}

bool arg_parser(int argc, char** argv, std::string &video, std::string &mode, std::string &color, std::string &modelPath, bool &debug) {
    // TODO: Add try-catch exception
    cxxopts::Options options("cpp_handtrack", "Hand Detection Tool (OpenCV/MediaPipe/ML)");
    options.add_options()
        ("m,mode", "Detection type: skin, contour, mediapipe, ml",
            cxxopts::value<std::string>()->default_value("skin"))
        ("c,color", "Detection type: hsv, ycrcb",
            cxxopts::value<std::string>()->default_value("hsv"))
        ("v,video", "Video source: camera index (e.g. 0), file path, or IP stream",
            cxxopts::value<std::string>()->default_value("0"))
        ("model", "Path to ML model (for ML mode)",
            cxxopts::value<std::string>()->default_value(""))
        ("debug", "Enable debug UI (window overlays, logging)",
            cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print usage");
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return true;
    }

    // Parse all values
    video = result["video"].as<std::string>();
    mode = result["mode"].as<std::string>();
    color = result["color"].as<std::string>();
    modelPath = result["model"].as<std::string>();
    debug = result["debug"].as<bool>();
    return true;
}


int main(int argc, char** argv) {

    // Argument parser
    std::string video, mode, color, modelPath;
    bool debug;
    if (!arg_parser(argc, argv, video, mode, color, modelPath, debug)) {
        return 1;
    }

    cv::VideoCapture cap(std::stoi(video));
    if (!cap.isOpened()) {
        std::cerr << "Camera not accessible\n";
        return -1;
    }

    // init skin detect
    ColorSpace color_space = ColorSpace::HSV;
    if (mode == "skin" || mode == "contour") {
        if (color == "ycrcb") {
            color_space = ColorSpace::YCrCb;
        }
        initSkinDetect(color_space);
    }

    SkinSegmenter skinSegmenter(color_space);
    ContourAnalyzer contourAnalyzer;
    // MediaPipeDetector mediaPipeDetector;
    // MLClassifier classifier("");

    while (true) {
        // FPS count
        auto lastTime = std::chrono::high_resolution_clock::now();
        float fps = 0.0f;

        cv::Mat frame, output;
        cap >> frame;
        if (frame.empty()) break;
        if (mode == "skin") {
            runSkinDetect(frame, skinSegmenter, color_space, output);
        } else if (mode == "contour") {
            runSkinDetect(frame, skinSegmenter, color_space, output);

            // Post-processing
            cv::Mat morph;
            cv::morphologyEx(output, morph, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 1);

            // Draw on original frame
            contourAnalyzer.analyzeHandContour(morph, frame, angleThreshold, depthThreshold, startRatioSlider*0.01f, farRatioSlider*0.01f);
        } else if (mode == "mediapipe") {
            // TODO
            // mediaPipeDetector.detect(frame, output);
            // cv::imshow("MediaPipe Output", output);
        } else if (mode == "ml") {
            // TODO
            // std::string result = classifier.classify(frame);  // placeholder
            // output = frame.clone();
            // drawLabel(output, result, {10, 30});
            // cv::imshow("ML Output", output);
        }

        // FPS label in top-left corner
        auto currentTime = std::chrono::high_resolution_clock::now();
        float delta = std::chrono::duration<float>(currentTime - lastTime).count();
        fps = 1.0f / delta;
        lastTime = currentTime;

        std::string fpsText = "FPS: " + std::to_string(int(fps));
        cv::putText(frame, fpsText, cv::Point(20, output.rows - 20),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);

        // Resize, label, and combine frame
        cv::Mat maskColor;
        cv::cvtColor(output, maskColor, cv::COLOR_GRAY2BGR);
        cv::resize(frame, frame, cv::Size(640, 480));
        cv::resize(maskColor, maskColor, cv::Size(640, 480));

        cv::putText(maskColor, (color_space == ColorSpace::YCrCb) ? "YCrCb" : "HSV", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 0.5);

        cv::Mat combined;
        cv::hconcat(frame, maskColor, combined);
        cv::imshow("Output", combined);

        if (cv::waitKey(1) == 27) break;  // ESC
    }

    return 0;
}
