#include <opencv2/opencv.hpp>

int main()
{
    // ������������ ����������.
    cv::Mat image = cv::imread("C:\\Users\\oleg\\Desktop\\1.png");

    // ������������ ���������� �� �������� �����
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // ����������� ���������� ��� ������������ ������ ������
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY_INV);

    // ������������ �������� ������� ��� ��������� ����
    cv::medianBlur(binary, binary, 11);

    // ����������� ������� �� ���������
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // ���� ���������� �������, ���� �� ����� 5 �����
    std::vector<cv::Point> wheelContour;
    double maxArea = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea && contours[i].size() >= 5)
        {
            maxArea = area;
            wheelContour = contours[i];
        }
    }

    // ��������� ������ ������ �� ���������
    cv::RotatedRect ellipse = cv::fitEllipse(wheelContour);
    cv::ellipse(image, ellipse, cv::Scalar(0, 255, 0), 2);

    // ����� ���������� � ��������� ������� ������
    cv::imshow("Wheel Detection", image);
    cv::waitKey(0);

    return 0;
}