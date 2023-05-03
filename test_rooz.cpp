#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;


//void circlePaint(Mat img, int cols,int  rows, int color) {
//    Point center(cols, rows);
//    int radius = 20;
//
//    circle(img, center, radius, color, 3);
//}
//
//void ellipsesPaint(Mat img, int cols, int  rows, int color) {
//    Point center(cols, rows);
//    Size axes(80, 30);
//    double angle = 10;
//
//    ellipse(img, center, axes, angle, 0, 360, color, -4);
//}
//
//void linePaint(Mat img, int cols, int rows, int colsStart, int  rowsEnd, int color) {
//    Point start(cols, rows);
//    Point end(colsStart, rowsEnd);
//
//    line(img, start, end, color, 2);
//}
//
//void rectanglePaint(Mat img, int cols, int  rows, int color) {
//    Point topleft(cols, rows);
//    Size size(rand() % 5+40, rand() % 5+40);
//    Point bottomright(topleft.x + size.width, topleft.y + size.height);
//
//
//    rectangle(img, topleft, bottomright, color, -1);
//}
//
//void textWriter(Mat img, string text) {
//    putText(img, text, cv::Point(40, 90), cv::FONT_HERSHEY_PLAIN, 1.0, 0, 2);
//    imshow("Text", img);
//}
//
//void imageInsert(Mat img, Mat logo) {
//    Mat imageROI(img, Rect(img.cols - logo.cols, img.rows - logo.rows, logo.cols, logo.rows));
//
//    logo.copyTo(imageROI);
//  
//}
//
//void imageInsertWhite(Mat img, Mat logo) {
//    Mat imageROI(img, Rect(img.cols - logo.cols, img.rows - logo.rows, logo.cols, logo.rows));
//
//  
//    Mat mask = Mat::zeros(logo.size(), CV_8UC1);
//    Scalar white(254, 254, 254);
//
//    inRange(logo, white, white, mask);
//
//    logo.copyTo(imageROI, mask);
//}
//
//
//
//void onMouse(int event, int x, int y, int flags, void* param) {
//    Mat* im = reinterpret_cast<Mat*>(param);
//    Vec3b intensity = im->at<Vec3b>(y, x);
//    int blue = intensity.val[0];
//    int green = intensity.val[1];
//    int red = intensity.val[2];
//    switch (event)
//    {
//    case EVENT_LBUTTONDOWN:
//        cout << "at (" << x << ", " << y << ") RGB value is: (" << red << ", " << green << ", " << blue << ")" << endl;
//        break;
//
//    }
//}

//int main()
//{
//    Mat img = imread("C:\\Users\\oleg\\Desktop\\oleg_1.bmp");
//    if (img.empty()){
//        cout << "No image";
//    }
//
//    namedWindow("Pixel IMREAD_GRAYSCALE");
//    imshow("Pixel IMREAD_GRAYSCALE", img);
//
//    setMouseCallback("Pixel IMREAD_GRAYSCALE", onMouse, reinterpret_cast<void*>(&img));
//
//    waitKey(0);
//
//    return 0;
//
//}

