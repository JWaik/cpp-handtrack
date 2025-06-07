#pragma once
#include <opencv2/opencv.hpp>

enum class ColorSpace { HSV, YCrCb };

struct ThresholdRange {
    int min1, min2, min3;
    int max1, max2, max3;
};

class SkinSegmenter {
public:
    SkinSegmenter(ColorSpace colorSpace = ColorSpace::HSV);
    void segment(const cv::Mat& input, cv::Mat& mask);

    void setThresholds(const ThresholdRange& range);

private:
    ColorSpace mode;
    ThresholdRange thresholds;

    void segmentHSV(const cv::Mat& input, cv::Mat& mask);
    void segmentYCrCb(const cv::Mat& input, cv::Mat& mask);
};
