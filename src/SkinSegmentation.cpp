#include "SkinSegmenter.hpp"

SkinSegmenter::SkinSegmenter(ColorSpace colorSpace) : mode(colorSpace) {
    thresholds = {0, 30, 60, 20, 150, 255}; // default HSV range
}

void SkinSegmenter::setThresholds(const ThresholdRange& range) {
    thresholds = range;
}

void SkinSegmenter::segment(const cv::Mat& input, cv::Mat& mask) {
    if (mode == ColorSpace::HSV) {
        segmentHSV(input, mask);
    } else {
        segmentYCrCb(input, mask);
    }
}

// HSV thresholding
void SkinSegmenter::segmentHSV(const cv::Mat& input, cv::Mat& mask) {
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv,
                cv::Scalar(thresholds.min1, thresholds.min2, thresholds.min3),
                cv::Scalar(thresholds.max1, thresholds.max2, thresholds.max3),
                mask);
    cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);
}


// YCrCb thresholding
void SkinSegmenter::segmentYCrCb(const cv::Mat& input, cv::Mat& mask) {
    cv::Mat ycrcb;
    cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);

    // Typical skin color range in YCrCb
    cv::inRange(ycrcb,
                cv::Scalar(thresholds.min1, thresholds.min2, thresholds.min3),
                cv::Scalar(thresholds.max1, thresholds.max2, thresholds.max3),
                mask);
    cv::GaussianBlur(mask, mask, cv::Size(3, 3), 0);
}
