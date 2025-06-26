#include "efficientGoodFeaturesToTrack.h"

vector<double> convolution(const cv::Mat& image, const vector<cv::KeyPoint>& pts, const cv::Mat& kernal)
{
    double pixSum = 0;
    vector<double> grad;
    for(int i = 0; i < pts.size(); i++)
    {
        int row = floor(pts[i].pt.y);
        int col = floor(pts[i].pt.x);
        //the pixel locates in the middle of the kernal 
        for (int k = 0; k < kernal.rows; k++) 
            for (int l = 0; l < kernal.cols; l++)
                pixSum += kernal.at<double>(k, l)*double(image.at<uchar>(k+row-1, l+col-1));
        grad.push_back(abs(pixSum));
        pixSum = 0;
    }

    return grad;
}


typedef pair<int, double> PAIR;
bool cmp_by_value(const PAIR& lhs, const PAIR& rhs)
{
    return lhs.second > rhs.second;
}


void duy_GoodFeaturesToTrack(InputArray _image, vector<cv::Point2f>& have_corners, vector<cv::Point2f>& corners, int maxCorners, double minDistance, cv::InputArray _mask)
{
    corners.clear();
    Mat image = _image.getMat();
    if(image.empty())
        return;
    
    //---------------- 1.ADD MASK
    Mat mask;
    if (!_mask.empty())
    {
        mask = _mask.getMat();
        if (mask.size() != image.size())
        {
            std::cerr << "Mask size must match image size!" << std::endl;
            return;
        }
    }
    else
    {
        mask = Mat(image.size(), CV_8UC1, Scalar(255)); // If no mask is provided, use a mask that allows all pixels
    }
    //--------------------END 

    vector<cv::KeyPoint> keypoints;
    cv::FAST(image, keypoints, 20, true);


    // ----------------------- 1.ADD MASK
    vector<cv::KeyPoint> keypoints_masked;
    for (const auto& kp : keypoints)
    {
        if (mask.at<uchar>(kp.pt) > 0)
        {
            keypoints_masked.push_back(kp);
        }
    }
    // ----------------------- END 


    if(keypoints.empty())
        return;

    cv::Mat kernal_x = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat kernal_y = (cv::Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    vector<double> grad_x;
    vector<double> grad_y;
    grad_x = convolution(image, keypoints, kernal_x);
    grad_y = convolution(image, keypoints, kernal_y);

    //2. minMaxLoc
    vector<pair<int, double>> eigens;
    for(int i = 0; i < grad_x.size(); i++)
    {
        Eigen::Matrix2d cov;
        cov(0, 0) = grad_x[i] * grad_x[i];
        cov(0, 1) = grad_x[i] * grad_y[i];
        cov(1, 0) = grad_x[i] * grad_y[i];
        cov(1, 1) = grad_y[i] * grad_y[i];

        EigenSolver<Matrix2d> es(cov);
        Eigen::Vector2cd eig_ = es.eigenvalues();
        Vector2d eig = eig_.real();
        double eg1 = eig(0);
        double eg2 = eig(1);
        if(eg1 >= eg2)
            eigens.push_back(make_pair(i, eg1));
        else
            eigens.push_back(make_pair(i, eg2));
        
    }

    sort(eigens.begin(), eigens.end(), cmp_by_value);
    vector<cv::KeyPoint> keypoints_;
    for(int i = 0; i < eigens.size(); i++)
        keypoints_.push_back(keypoints[eigens[i].first]);

    if(minDistance >= 1)
    {
        int ncorners = 0;
        int w = image.cols;
        int h = image.rows;

        const int cell_size = cvRound(minDistance);
        const int grid_width = (w + cell_size -1) / cell_size;
        const int grid_height = (h + cell_size -1) / cell_size;

        std::vector<std::vector<cv::Point2f>> grid(grid_width * grid_height);

        minDistance *= minDistance;
        //push the already exist feature points into the grid
        for(int i = 0; i < have_corners.size(); i++)
        {
            int y = (int)(have_corners[i].y);
            int x = (int)(have_corners[i].x);

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            if(x_cell <= grid_width && y_cell <= grid_height)
                grid[y_cell*grid_width + x_cell].push_back(have_corners[i]);

            corners.push_back(have_corners[i]);
            ++ncorners;
        }

        for(int i = 0; i < keypoints_.size(); i++)
        {
            if(keypoints_[i].pt.y < 0 || keypoints_[i].pt.y > image.rows - 1)
                continue;
            if(keypoints_[i].pt.x < 0 || keypoints_[i].pt.x > image.cols - 1)
                continue;    
            int y = (int)(keypoints_[i].pt.y);
            int x = (int)(keypoints_[i].pt.x);

            bool good = true;

            int x_cell = x / cell_size;
            int y_cell = y / cell_size;

            int x1 = x_cell -1;
            int y1 = y_cell -1;
            int x2 = x_cell +1;
            int y2 = y_cell +1;

            //boundary check
            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(grid_width-1, x2);
            y2 = std::min(grid_height-1, y2);

            //select feature points satisfy minDistance threshold
            for(int yy = y1; yy <= y2; yy++)
            {
                for(int xx = x1; xx <= x2; xx++)
                {
                    std::vector<cv::Point2f>& m = grid[yy*grid_width+xx];

                    if(m.size())
                    {
                        for(int j = 0; j < m.size(); j++)
                        {
                            float dx = x - m[j].x;
                            float dy = y - m[j].y;

                            if(dx*dx + dy*dy < minDistance)
                            {
                                good = false;
                                goto break_out;
                            }
                        }
                    }
                }
            }

            break_out:

            if(good)
            {
                grid[y_cell*grid_width + x_cell].push_back(cv::Point2f((float)x, (float)y));

                corners.push_back(cv::Point2f((float)x, (float)y));
                ++ncorners;

                if(maxCorners > 0 && (int)ncorners == maxCorners)
                    break;
            }
        }

    }
    else
    {
        return;
    }

    if(have_corners.size() != 0)
        corners.erase(corners.begin(), corners.end()+ have_corners.size()); 
}