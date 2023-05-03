#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <random>
#include <ctime>

using namespace std;
using namespace cv;



class ContentFinder {
private:
    float hranges[2];
    const float* ranges[3];
    int channels[3];

    float threshold;

    cv::MatND histogram;

public:
    ContentFinder() : threshold(-1.0f) {
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
    }

    void setHistogram(const cv::MatND& h) {
        histogram = h;
        cv::normalize(histogram, histogram, 1.0);
    }

    void setThreshold(float t) {
        threshold = t;
    }

    cv::Mat find(const cv::Mat& image) {
        cv::Mat result;
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;

        result = find(image, hranges[0], hranges[1], channels);

        return result;
    }

    cv::Mat find(const cv::Mat& image, float minValue, float maxValue, int* channels) {
        cv::Mat result;
        hranges[0] = minValue;
        hranges[1] = maxValue;

        cv::calcBackProject(&image,
            1,
            channels,
            histogram,
            result,
            ranges,
            55.0);

        if (threshold > 0.0) {
            cv::threshold(result, result, 255.0 * threshold, 255.0, cv::THRESH_BINARY);
        }

        cv::Mat neg_result;
        cv::bitwise_not(result, neg_result);

        return neg_result;
    }
};

class ColorHistogram
{
private:
    int histSize[3];
    float hranges[2];
    const float* ranges[3];
    int channels[3];

public:
    ColorHistogram() : histSize{ 8, 8, 8 }
    {
        hranges[0] = 0.0;
        hranges[1] = 256.0;
        ranges[0] = hranges;
        ranges[1] = hranges;
        ranges[2] = hranges;
        channels[0] = 0;
        channels[1] = 1;
        channels[2] = 2;
    }

    void setSize(int size)
    {
        histSize[0] = histSize[1] = histSize[2] = size;
    }

    cv::Mat getHistogram(const cv::Mat& image)
    {
        cv::Mat hist;

        cv::calcHist(&image,
            1,
            channels,
            cv::Mat(),
            hist,
            3,
            histSize,
            ranges);

        return hist;
    }

    cv::Mat getHueHistogram(const cv::Mat& image, int minSaturation)
    {
        cv::Mat hsv;
        cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        if (minSaturation > 0)
        {
            std::vector<cv::Mat> v;
            cv::split(hsv, v);

            cv::threshold(v[1], mask, minSaturation, 255, cv::THRESH_BINARY);
        }

        cv::Mat hist;
        int hrange[] = { 0, 180 };
        const float* ranges[] = { (const float*)hrange };
        int channels[] = { 0 };

        cv::calcHist(&hsv,
            1,
            channels,
            mask,
            hist,
            1,
            &histSize[0],
            ranges);

        return hist;
    }
};


class Histogram1D
{
private:
    int nbins;
public:
    void setNBins(int n) { nbins = n; }
public:

    static cv::Mat equalize(const cv::Mat& image)
    {
        cv::Mat result;
        cv::equalizeHist(image, result);
        return result;
    }

    cv::Mat getHistogram(const cv::Mat& image)
    {
        cv::Mat hist;
        int histSize = nbins;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        cv::calcHist(&image, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, true, false);
        return hist;
    }

    cv::Mat getHistogramImage(const cv::Mat& image)
    {
        cv::Mat hist = getHistogram(image);

        double maxVal = 0, minVal = 0;
        cv::minMaxLoc(hist, &minVal, &maxVal, 0, 0);

        int histSize = hist.rows;
        cv::Mat histImg(histSize, nbins, CV_8U, cv::Scalar(255));

        for (int h = 0; h < histSize; h++)
        {
            float binVal = hist.at<float>(h);
            int intensity = static_cast<int>(binVal * nbins / maxVal);
            cv::line(histImg, cv::Point(0, histSize - h), cv::Point(intensity, histSize - h), cv::Scalar(0), 1);
        }

        return histImg;
    }

    cv::Mat applyLookUp(const cv::Mat& image, const cv::Mat& lookup)
    {
        cv::Mat result;
        cv::LUT(image, lookup, result);
        return result;
    }

    cv::Mat stretch(const cv::Mat& image, float percentile)
    {
        float number = image.total() * percentile;

        cv::Mat hist = getHistogram(image);

        int imin = 0;
        for (float count = 0.0; imin < 256; imin++)
        {
            if ((count += hist.at<float>(imin)) >= number)
            {
                break;
            }
        }

        int imax = 255;
        for (float count = 0.0; imax >= 0; imax--)
        {
            if ((count += hist.at<float>(imax)) >= number)
            {
                break;
            }
        }

        int dims[1] = { 256 };
        cv::Mat lookup(1, dims, CV_8U);

        for (int i = 0; i < 256; i++)
        {
            if (i < imin)
            {
                lookup.at<uchar>(i) = 0;
            }
            else if (i > imax)
            {
                lookup.at<uchar>(i) = 255;
            }
            else
            {
                lookup.at<uchar>(i) = cvRound(255.0 * (i - imin) / (imax - imin));
            }
        }

        cv::Mat result;
        result = applyLookUp(image, lookup);

        return result;
    }
};



//int main()
//{
//    cv::Mat image = cv::imread("C:\\Users\\oleg\\Desktop\\1.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat object = cv::imread("C:\\Users\\oleg\\Desktop\\2.jpg", cv::IMREAD_GRAYSCALE);
//
//    cv::Mat hist_object, hist_image;
//    int histSize = 256;
//    float range[] = { 0, 256 };
//    const float* histRange = { range };
//    cv::calcHist(&object, 1, 0, cv::Mat(), hist_object, 1, &histSize, &histRange);
//    cv::normalize(hist_object, hist_object, 1, 0, cv::NORM_L1);
//
//    cv::namedWindow("Object Histogram", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Object Histogram", hist_object);
//
//    cv::Mat result;
//    cv::matchTemplate(image, object, result, cv::TM_CCOEFF_NORMED);
//    cv::normalize(result, result, 1, 0, cv::NORM_L1);
//
//    double minVal, maxVal;
//    cv::Point minLoc, maxLoc;
//    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
//
//    cv::Rect roi(maxLoc.x, maxLoc.y, object.cols, object.rows);
//    cv::rectangle(image, roi, cv::Scalar(255), 2);
//
//    cv::namedWindow("Tracking Result", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Tracking Result", image);
//
//    cv::waitKey(0);
//    return 0;
//}
