#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <random>
#include <ctime>

using namespace std;
using namespace cv;

class CDetector {
private:
    int mDist;
    cv::Vec3b tColor;
    cv::Mat conv;
    bool isLAB;
    cv::Mat rslt;

public:
    CDetector() : mDist(100), tColor(0, 0, 0), isLAB(false) { }

    CDetector(bool isLAB) : mDist(100), tColor(0, 0, 0), isLAB(isLAB) { }

    CDetector(uchar blue, uchar green, uchar red, int mxDist = 100, bool isLAB = false) : mDist(mxDist), isLAB(isLAB)
    {
        setTargetColor(blue, green, red);
    }

    int getDistToTargetColor(const cv::Vec3b& color) const
    {
        return getColorDistance(color, tColor);
    }

    int getColorDistance(const cv::Vec3b& color1, const cv::Vec3b& color2) const
    {
        return abs(color1[0] - color2[0]) +
            abs(color1[1] - color2[1]) +
            abs(color1[2] - color2[2]);
    }

    cv::Mat proc(const cv::Mat& img)
    {
        rslt.create(img.size(), CV_8U);

        if (isLAB)
        {
            cv::cvtColor(img, conv, cv::COLOR_BGR2Lab);
        }

        cv::Mat_<cv::Vec3b>::const_iterator itr = img.begin<cv::Vec3b>();
        cv::Mat_<cv::Vec3b>::const_iterator itrEnd = img.end<cv::Vec3b>();
        cv::Mat_<uchar>::iterator itrOut = rslt.begin<uchar>();

        if (isLAB)
        {
            itr = conv.begin<cv::Vec3b>();
            itrEnd = conv.end<cv::Vec3b>();
        }

        for (; itr != itrEnd; ++itr, ++itrOut)
        {
            if (getDistToTargetColor(*itr) < mDist)
            {
                *itrOut = 255;
            }
            else
            {
                *itrOut = 0;
            }
        }

        return rslt;
    }

    cv::Mat operator()(const cv::Mat& image)
    {
        cv::Mat input;

        if (isLAB)
        {
            cv::cvtColor(image, input, cv::COLOR_BGR2Lab);
        }
        else
        {
            input = image;
        }

        cv::Mat output;
        cv::absdiff(input, cv::Scalar(tColor), output);

        std::vector<cv::Mat> images;
        cv::split(output, images);

        output = images[0] + images[1] + images[2];
        cv::threshold(output, output, mDist, 255, cv::THRESH_BINARY_INV);

        return output;
    }

    void setColorDistanceThreshold(int distance)
    {
        if (distance < 0)
        {
            distance = 0;
        }
        mDist = distance;
    }

    int getColorDistanceThreshold() const
    {
        return mDist;
    }

    void setTargetColor(uchar blue, uchar green, uchar red)
    {
        tColor = cv::Vec3b(blue, green, red);

        if (isLAB)
        {
            cv::Mat tmp(1, 1, CV_8UC3);
            tmp.at<cv::Vec3b>(0, 0) = cv::Vec3b(blue, green, red);

            cv::cvtColor(tmp, tmp, cv::COLOR_BGR2Lab);

            tColor = tmp.at<cv::Vec3b>(0, 0);
        }
    }

    void setTargetColor(cv::Vec3b color)
    {
        tColor = color;
    }

    cv::Vec3b getTargetColor() const
    {
        return tColor;
    }
};

void detectHScolor(const cv::Mat& image, double minHue, double maxHue, double minSat, double maxSat, cv::Mat& mask)
{
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    cv::Mat mask1;
    cv::threshold(channels[0], mask1, maxHue, 255, cv::THRESH_BINARY_INV);
    cv::Mat mask2;
    cv::threshold(channels[0], mask2, minHue, 255, cv::THRESH_BINARY);

    cv::Mat hueMask;
    if (minHue < maxHue)
    {
        hueMask = mask1 & mask2;
    }
    else
    {
        hueMask = mask1 | mask2;
    }

    cv::Mat satMask;
    cv::inRange(channels[1], minSat, maxSat, satMask);

    mask = hueMask & satMask;
}


//int main()
//{
//
//    cv::Mat image = cv::imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp");
//    if (image.empty())
//    {
//        return 0;
//    }
//    cv::namedWindow("Image");
//    cv::imshow("Image", image);
//    cv::Mat mask;
//    detectHScolor(image, 160, 10, 25, 166, mask);
//
//    cv::Mat detected(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
//    image.copyTo(detected, mask);
//    cv::imshow("Result", detected);
//
//    cv::waitKey();
//
//    return 0;
//}



