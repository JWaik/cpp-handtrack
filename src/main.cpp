#include <opencv2/opencv.hpp>
#include "SkinSegmenter.hpp"
#include "ContourAnalyzer.hpp"
#include "MediaPipeDetector.hpp"
#include "MLClassifier.hpp"
#include "Utils.hpp"

// HSV thresholds
int minH = 0, minS = 30, minV = 60;
int maxH = 20, maxS = 150, maxV = 255;

// YCrCb thresholds
int minY = 0, minCr = 133, minCb = 77;
int maxY = 255, maxCr = 173, maxCb = 127;

void setupTrackbarsHSV() {
    cv::namedWindow("ControlsHSV", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Min H", "ControlsHSV", &minH, 179);
    cv::createTrackbar("Max H", "ControlsHSV", &maxH, 179);
    cv::createTrackbar("Min S", "ControlsHSV", &minS, 255);
    cv::createTrackbar("Max S", "ControlsHSV", &maxS, 255);
    cv::createTrackbar("Min V", "ControlsHSV", &minV, 255);
    cv::createTrackbar("Max V", "ControlsHSV", &maxV, 255);
}

void setupTrackbarsYCrCb() {
    cv::namedWindow("ControlsYCrCb", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Min Y", "ControlsYCrCb", &minY, 255);
    cv::createTrackbar("Max Y", "ControlsYCrCb", &maxY, 255);
    cv::createTrackbar("Min Cr", "ControlsYCrCb", &minCr, 255);
    cv::createTrackbar("Max Cr", "ControlsYCrCb", &maxCr, 255);
    cv::createTrackbar("Min Cb", "ControlsYCrCb", &minCb, 255);
    cv::createTrackbar("Max Cb", "ControlsYCrCb", &maxCb, 255);
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

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "skin";
    std::string color_mode = argc > 2 ? argv[2] : "hsv";

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera not accessible\n";
        return -1;
    }

    // init skin detect
    ColorSpace color_space = ColorSpace::HSV;
    if (mode == "skin" || mode == "contour") {
        if (color_mode == "ycrcb") {
            color_space = ColorSpace::YCrCb;
        }
        initSkinDetect(color_space);
    }

    SkinSegmenter skinSegmenter(color_space);
    // TODO
    // ContourAnalyzer contourAnalyzer;
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
            // TODO
            // skinSegmenter.segment(frame, mask);
            // contourAnalyzer.analyze(mask, output);
            // cv::imshow("Contour Output", output);
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

        cv::putText(frame, "Original", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 0.5);
        cv::putText(maskColor, (color_space == ColorSpace::YCrCb) ? "YCrCb" : "HSV", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 0.5);

        cv::Mat combined;
        cv::hconcat(frame, maskColor, combined);
        cv::imshow("Output", combined);

        if (cv::waitKey(1) == 27) break;  // ESC
    }

    return 0;
}
