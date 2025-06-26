#pragma once 

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>
#include <eigen3/Eigen/Dense>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;
using namespace std::chrono;

vector<double> convolution(const cv::Mat& image, const vector<cv::KeyPoint>& pts, const cv::Mat& kernal);

typedef pair<int, double> PAIR;

bool cmp_by_value(const PAIR& lhs, const PAIR& rhs);

// void duy_GoodFeaturesToTrack(InputArray _image, vector<cv::Point2f>& have_corners, vector<cv::Point2f>& corners, int maxCorners, double minDistance);
void duy_GoodFeaturesToTrack(cv::InputArray _image, std::vector<cv::Point2f>& have_corners, std::vector<cv::Point2f>& corners, int maxCorners, double minDistance, cv::InputArray _mask = cv::noArray());