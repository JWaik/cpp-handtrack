#pragma once
#include <opencv2/opencv.hpp>

class MediaPipeDetector {
public:
    MediaPipeDetector();
    bool detect(const cv::Mat& frame, cv::Mat& output);  // stub
};