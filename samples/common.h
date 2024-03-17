#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>

#include "detection.h"

using namespace radar;

std::vector<std::string> kArmorNames{"B1", "B2", "B3", "B4", "B5", "R1",
                                     "R2", "R3", "R4", "R5", "Bs", "Rs"};

inline cv::Rect2f cv_rect(const Detection& detection) {
    return cv::Rect2f(detection.x, detection.y, detection.width,
                      detection.height);
}

inline void drawDetection(cv::Mat& image, std::string_view text,
                          const cv::Rect2f& rect, const cv::Scalar& color,
                          int line_width = 2, float font_scale = 1,
                          int font_thickness = 4) {
    cv::rectangle(image, rect, color, line_width);

    int base_line = 0;
    auto label_size = cv::getTextSize(text.data(), cv::FONT_HERSHEY_SIMPLEX,
                                      font_scale, font_thickness, &base_line);
    cv::Rect2f label_rect(rect.x, rect.y - label_size.height - base_line / 2,
                          label_size.width, label_size.height + base_line);
    cv::rectangle(image, label_rect, color, -1);
    cv::putText(image, text.data(), cv::Point(rect.x, rect.y),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255),
                font_thickness);
}