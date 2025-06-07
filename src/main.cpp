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

void initSkinDetect() {
    setupTrackbarsHSV();
    setupTrackbarsYCrCb();
}

void runSkinDetect(cv::Mat &frame, SkinSegmenter &segSkinHsv, SkinSegmenter &segSkinYcrcb, cv::Mat &output) {
    cv::Mat mask_hsv, mask_ycrcb;

    ThresholdRange trHSV, trYCrCb;
    trHSV = {minH, minS, minV, maxH, maxS, maxV};
    trYCrCb = {minY, minCr, minCb, maxY, maxCr, maxCb};

    segSkinHsv.setThresholds(trHSV);
    segSkinHsv.segment(frame, mask_hsv);
    segSkinYcrcb.setThresholds(trYCrCb);
    segSkinYcrcb.segment(frame, mask_ycrcb);

    cv::Mat maskColorHsv, maskColorYcrcb;
    cv::cvtColor(mask_hsv, maskColorHsv, cv::COLOR_GRAY2BGR);
    cv::cvtColor(mask_ycrcb, maskColorYcrcb, cv::COLOR_GRAY2BGR);

    cv::resize(frame, frame, cv::Size(320, 240));
    cv::resize(maskColorHsv, maskColorHsv, cv::Size(320, 240));
    cv::resize(maskColorYcrcb, maskColorYcrcb, cv::Size(320, 240));

    cv::Mat combinedHsv;
    cv::hconcat(frame, maskColorHsv, combinedHsv);  // Combine side by side
    cv::hconcat(combinedHsv, maskColorYcrcb, output);  // Combine side by side

}

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "skin";

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera not accessible\n";
        return -1;
    }

    // init
    if (mode == "skin") {
        initSkinDetect();
    }

    SkinSegmenter skinSegmenterHSV(ColorSpace::HSV);
    SkinSegmenter skinSegmenterYCrCb(ColorSpace::YCrCb);
    // TODO
    // ContourAnalyzer contourAnalyzer;
    // MediaPipeDetector mediaPipeDetector;
    // MLClassifier classifier("");

    while (true) {
        cv::Mat frame, output;
        cap >> frame;
        if (frame.empty()) break;
        if (mode == "skin") {
            runSkinDetect(frame, skinSegmenterHSV, skinSegmenterYCrCb, output);
            cv::imshow("Skin Detection", output);
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

        if (cv::waitKey(1) == 27) break;  // ESC
    }

    return 0;
}
