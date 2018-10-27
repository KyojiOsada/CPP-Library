#include <iostream>
#include "opencv2\\opencv.hpp"
#include "opencv2\\opencv_lib.hpp"

const int tempThres = 48;
const int srcThres  = 36;
const double huMomentThres = 0.01;

int main()
{
    cv::Mat temp = cv::imread("source1.jpg", 0);
    if (temp.empty()){
        std::cout << "Read Error" << std::endl;
        return -1;
    }

    cv::Mat src = cv::imread("source2.jpg");
    if (src.empty()){
        std::cout << "Read Error" << std::endl;
        return -1;
    }
    cv::Mat src_gray;
    cv::cvtColor(src, src_gray, CV_BGR2GRAY);

    cv::Mat temp_bin, src_bin;
    cv::threshold(temp, temp_bin, tempThres, 255, cv::THRESH_BINARY_INV);
    cv::threshold(src_gray, src_bin, srcThres, 255, cv::THRESH_BINARY_INV);

    cv::morphologyEx(temp_bin, temp_bin, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);
    cv::morphologyEx(src_bin,  src_bin,  cv::MORPH_OPEN, cv::Mat(), cv::Point(-1, -1), 2);

    cv::Mat labelsImg;
    cv::Mat stats;
    cv::Mat centroids;
    int nLabels = cv::connectedComponentsWithStats(src_bin, labelsImg, stats, centroids);

    cv::Mat roiImg;
    cv::cvtColor(src_bin, roiImg, CV_GRAY2BGR);
    std::vector<cv::Rect> roiRects;
    for (int i = 1; i < nLabels; i++) {
        int *param = stats.ptr<int>(i);

        int x = param[cv::ConnectedComponentsTypes::CC_STAT_LEFT];
        int y = param[cv::ConnectedComponentsTypes::CC_STAT_TOP];
        int height = param[cv::ConnectedComponentsTypes::CC_STAT_HEIGHT];
        int width = param[cv::ConnectedComponentsTypes::CC_STAT_WIDTH];
        roiRects.push_back(cv::Rect(x, y, width, height));

        cv::rectangle(roiImg, roiRects.at(i-1), cv::Scalar(0, 255, 0), 2);
    }

    cv::Mat dst = src.clone();
    for (int i = 1; i < nLabels; i++){
        cv::Mat roi = src_bin(roiRects.at(i-1));
        double similarity = cv::matchShapes(temp_bin, roi, CV_CONTOURS_MATCH_I1, 0);

        if (similarity < huMomentThres){
            cv::rectangle(dst, roiRects.at(i - 1), cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("template", temp);
    cv::imshow("src", src);
    cv::imshow("dst", dst);
    cv::waitKey();

    cv::imwrite("dst.jpg", dst);

    return 0;
}
