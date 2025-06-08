#include "ContourAnalyzer.hpp"


void ContourAnalyzer::analyzeHandContour(const cv::Mat& mask, cv::Mat& output, int angleTres, int depthTres, float startRatio, float farRatio) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return;

    // Find the largest contour (assumed to be the hand)
    size_t largestIdx = 0;
    double maxArea = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            largestIdx = i;
        }
    }

    const std::vector<cv::Point>& handContour = contours[largestIdx];

    // Convex hull
    std::vector<int> hullIndices;
    cv::convexHull(handContour, hullIndices, false, false);

    std::vector<cv::Point> hullPoints;
    for (int idx : hullIndices)
        hullPoints.push_back(handContour[idx]);

    cv::drawContours(output, std::vector<std::vector<cv::Point>>{hullPoints}, 0, cv::Scalar(0, 255, 0), 2);

    // Convexity defects
    if (hullIndices.size() > 3) {
        std::vector<cv::Vec4i> defects;
        cv::convexityDefects(handContour, hullIndices, defects);

        // Calculate hand center and palm radius
        cv::Point2f center;
        float palmRadius;
        cv::minEnclosingCircle(handContour, center, palmRadius);
        cv::circle(output, center, (int)palmRadius, cv::Scalar(255, 255, 0), 1);  // Draw palm circle

        int fingerCount = 0;
        for (const auto& d : defects) {
            cv::Point ptStart = handContour[d[0]];
            cv::Point ptEnd   = handContour[d[1]];
            cv::Point ptFar   = handContour[d[2]];
            float depth       = d[3] / 256.0;

            // Basic rule to filter defects between fingers
            double a = cv::norm(ptStart - ptEnd);
            double b = cv::norm(ptStart - ptFar);
            double c = cv::norm(ptEnd   - ptFar);
            double angle = acos((b*b + c*c - a*a) / (2*b*c)) * 180 / CV_PI;
            if (angle < angleTres && depth > depthTres) {
                // Check distances from start/far/end to hand center
                double dStart = cv::norm(cv::Point2f(ptStart) - center);
                double dFar   = cv::norm(cv::Point2f(ptFar) - center);
                if (dStart > startRatio * palmRadius && dFar > farRatio * palmRadius) {
                    fingerCount++;
                    cv::circle(output, ptFar, 5, cv::Scalar(255, 0, 0), -1);
                }
            }
        }

        // Add 1 to count the thumb (hand typically makes n-1 defects)
        fingerCount = std::min(fingerCount + 1, 5);

        // Show result
        std::string text = "Fingers: " + std::to_string(fingerCount);
        cv::putText(output, text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
    }
}