#pragma once
#include <opencv2/opencv.hpp>

class ContourAnalyzer {
public:
    void analyze(const cv::Mat& binaryMask, cv::Mat& output);
};