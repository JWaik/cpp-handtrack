#pragma once
#include <opencv2/opencv.hpp>

class MLClassifier {
public:
    MLClassifier(const std::string& modelPath);
    std::string classify(const cv::Mat& roi);

private:
    // placeholder for SVM/CNN model loading
};