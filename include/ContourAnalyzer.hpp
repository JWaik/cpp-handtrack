#pragma once
#include <opencv2/opencv.hpp>

class ContourAnalyzer {
public:
    void analyzeHandContour(const cv::Mat& mask, cv::Mat& output);
};