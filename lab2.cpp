#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <random>
#include <ctime>

using namespace std;
using namespace cv;

void salt(Mat image, int n)
{
   

    int i, j;
    for (int k = 0; k < n; k++)
    {
       random_device rd;
       default_random_engine generator(rd());
       uniform_int_distribution<int> randomRow(0, image.rows - 1);
       uniform_int_distribution<int> randomCol(0, image.cols - 1);

      i = randomCol(generator);
      j = randomRow(generator);

        if (image.type() == CV_8UC1)
        {
            image.at<uchar>(j, i) = 255;
        }
        else if (image.type() == CV_8UC3)
        {
          
            image.at<cv::Vec3b>(j, i)[0] = 255;
            image.at<cv::Vec3b>(j, i)[1] = 255;
            image.at<cv::Vec3b>(j, i)[2] = 255;
        }
      
    }
    
}

void pepper(cv::Mat image, int n)
{
    random_device rd;
    default_random_engine generator(rd());
    std::uniform_int_distribution<int> randomRow(0, image.rows - 1);
    std::uniform_int_distribution<int> randomCol(0, image.cols - 1);

    int i, j;
    for (int k = 0; k < n; k++)
    {
        i = randomCol(generator);
        j = randomRow(generator);

        if (image.type() == CV_8UC1)
        {
            image.at<uchar>(j, i) = 0;
        }

        else if (image.type() == CV_8UC3)
        {

            image.at<cv::Vec3b>(j, i)[0] = 0;
            image.at<cv::Vec3b>(j, i)[1] = 0;
            image.at<cv::Vec3b>(j, i)[2] = 0;
        }
    }
}

void colorReduceIterators(cv::Mat image, int div = 64)
{
    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);

    uchar mask = 0xFF << n;
    uchar div2 = div >> 1; 

    
    cv::Mat_<cv::Vec3b>::iterator it = image.begin<cv::Vec3b>();
    cv::Mat_<cv::Vec3b>::iterator itend = image.end<cv::Vec3b>();

    for (; it != itend; ++it)
    {
        (*it)[0] &= mask;
        (*it)[0] += div2;
        (*it)[1] &= mask;
        (*it)[1] += div2;
        (*it)[2] &= mask;
        (*it)[2] += div2;
    }
}

void colorReduceAt(cv::Mat image, int div = 64)
{
    int nl = image.rows;
    int nc = image.cols;

    for (int j = 0; j < nl; j++)
    {
        for (int i = 0; i < nc; i++)
        {
            image.at<cv::Vec3b>(j, i)[0] = image.at<cv::Vec3b>(j, i)[0] / div * div + div / 2;
            image.at<cv::Vec3b>(j, i)[1] = image.at<cv::Vec3b>(j, i)[1] / div * div + div / 2;
            image.at<cv::Vec3b>(j, i)[2] = image.at<cv::Vec3b>(j, i)[2] / div * div + div / 2;
        }
    }
}

void colorReduce(cv::Mat image, int div = 64)
{
    int nl = image.rows;

    int nc = image.cols * image.channels();
    for (int j = 0; j < nl; j++)
    {
        uchar* data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++)
        {
            data[i] = data[i] / div * div + div / 2;
        }
    }
}

void colorReduceOneLoop(cv::Mat image, int div = 64)
{
    int nl = image.rows;
    int nc = image.cols * image.channels();

    if (image.isContinuous())
    {
        nc = nc * nl;
        nl = 1;
    }

    int n = static_cast<int>(log(static_cast<double>(div)) / log(2.0) + 0.5);
    uchar mask = 0xFF << n;
    uchar div2 = div >> 1;

    for (int j = 0; j < nl; j++)
    {
        uchar* data = image.ptr<uchar>(j);
        for (int i = 0; i < nc; i++)
        {
            *data &= mask;
            *data++ += div2;
        }
    }
}

void colorReduceNeighbourAccess(const cv::Mat& image, cv::Mat& result, int div = 64)
{
    result.create(image.size(), image.type());
    int nchannels = image.channels();
    for (int j = 1; j < image.rows - 1; j++)
    {
        const uchar* previous = image.ptr<const uchar>(j - 1);
        const uchar* current = image.ptr<const uchar>(j);
        const uchar* next = image.ptr<const uchar>(j + 1);

        uchar* previous_out = result.ptr<uchar>(j - 1);
        uchar* current_out = result.ptr<uchar>(j);
        uchar* next_out = result.ptr<uchar>(j + 1);

        for (int i = nchannels; i < (image.cols - 1) * nchannels; i++)
        {
            current_out[i] = current[i] / div * div + div / 2;
            current_out[i - nchannels] = current[i - nchannels] / div * div + div / 2;
            current_out[i + nchannels] = current[i + nchannels] / div * div + div / 2;
            previous_out[i] = previous[i] / div * div + div / 2;
            next_out[i] = next[i] / div * div + div / 2;
        }
    }

    result.row(0).setTo(cv::Scalar(0, 0, 0));
    result.row(result.rows - 1).setTo(cv::Scalar(0, 0, 0));
    result.col(0).setTo(cv::Scalar(0, 0, 0));
    result.col(result.cols - 1).setTo(cv::Scalar(0, 0, 0));
}


void sharpen(const cv::Mat& img, cv::Mat& result)
{
    result.create(img.size(), img.type());
    int nchannels = img.channels();
    for (int j = 1; j < img.rows - 1; j++)
    {
        const uchar* previous = img.ptr<const uchar>(j - 1);
        const uchar* current = img.ptr<const uchar>(j);
        const uchar* next = img.ptr<const uchar>(j + 1);
        uchar* output = result.ptr<uchar>(j);

        for (int i = nchannels; i < (img.cols - 1) * nchannels; i++)
        {
            *output++ = cv::saturate_cast<uchar>(5 * current[i] - current[i - nchannels] - current[i + nchannels] - previous[i] - next[i]);
        }
    }

    result.row(0).setTo(cv::Scalar(0, 0, 0));
    result.row(result.rows - 1).setTo(cv::Scalar(0));
    result.col(0).setTo(cv::Scalar(0, 0, 0));
    result.col(result.cols - 1).setTo(cv::Scalar(0));
}

void sharpen2D(const Mat& img, Mat& result)
{
    Mat kernel(3, 3, CV_32F, cv::Scalar(0));

    kernel.at<float>(1, 1) = 5.0;
    kernel.at<float>(0, 1) = -1.0;
    kernel.at<float>(2, 1) = -1.0;
    kernel.at<float>(1, 0) = -1.0;
    kernel.at<float>(1, 2) = -1.0;

    filter2D(img, result, img.depth(), kernel);
}

void addImages()
{
    cv::Mat image1 = cv::imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp");
    cv::Mat image2 = cv::imread("C:\\Users\\oleg\\Desktop\\love2.png");
    cv::Mat result = 0.5 * image1 + 0.5 * image2;

    cv::namedWindow("Img");
    cv::imshow("Img", result);
}

void ToBlueChannel()
{
    cv::Mat image1 = cv::imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp");

    cv::Mat image2 = cv::imread("C:\\Users\\oleg\\Desktop\\rain1.png", cv::IMREAD_GRAYSCALE), result;

    std::vector<cv::Mat> rain;
    cv::split(image1, rain);
    rain[1] += image2;

    cv::merge(rain, result);

    cv::namedWindow("Img");
    cv::imshow("Img", result);
}

void wave(const cv::Mat& image, cv::Mat& result)
{

    cv::Mat srcX(image.rows, image.cols, CV_32F);
    cv::Mat srcY(image.rows, image.cols, CV_32F);

    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            srcX.at<float>(i, j) = j;
            srcY.at<float>(i, j) = i + 5 * sin(j / 10.0);
        }
    }

    cv::remap(image,
        result,
        srcX,
        srcY,
        cv::INTER_LINEAR);
}

//int main()
//{
//    Mat img = imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp", IMREAD_GRAYSCALE);
//        if (img.empty()){
//            cout << "No image";
//        }
//  
//        Mat result;
//        wave(img, result);
//
//        namedWindow("Image");
//        imshow("Image1", result);
//    
//    
//        waitKey(0);
//    
//        return 0;
//
//}

//int main()
//{
//    cv::Mat img = cv::imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp", cv::IMREAD_GRAYSCALE);
//
//    cv::Mat sobelx, sobely;
//    cv::Sobel(img, sobelx, CV_64F, 1, 0, 5);
//    cv::Sobel(img, sobely, CV_64F, 0, 1, 5);
//
//    cv::Mat sobel = cv::Mat::zeros(sobelx.size(), sobelx.type());
//    cv::sqrt(sobelx.mul(sobelx) + sobely.mul(sobely), sobel);
//
//    cv::Mat pencil;
//    cv::normalize(sobel, pencil, 0, 255, cv::NORM_MINMAX, CV_8U);
//
//    cv::imshow("Pencil Sketch", pencil);
//    cv::waitKey(0);
//
//    return 0;
//}

void sobel_filter(const Mat& src, Mat& dst, int ksize, double scale, double delta)
{
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;

    Sobel(src, grad_x, CV_16S, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);

    Sobel(src, grad_y, CV_16S, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst);
}

//int main()
//{
//    Mat image = imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp", IMREAD_GRAYSCALE);
//    if (image.empty()) {
//        cout << "Could not open or find the image" << endl;
//        return -1;
//    }
//
//    Mat dst_5, dst_7, dst_9;
//
//    // testing Sobel filter with different kernel sizes
//    sobel_filter(image, dst_5, 5, 1, 0);
//    sobel_filter(image, dst_7, 7, 1, 0);
//    sobel_filter(image, dst_9, 9, 1, 0);
//
//    // display results
//    imshow("Original Image", image);
//    imshow("Sobel Filter (Kernel Size 5)", dst_5);
//    imshow("Sobel Filter (Kernel Size 7)", dst_7);
//    imshow("Sobel Filter (Kernel Size 9)", dst_9);
//
//    waitKey(0);
//    return 0;
//}