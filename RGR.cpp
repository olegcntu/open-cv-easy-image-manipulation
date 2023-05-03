#include <opencv2/opencv.hpp>

int main()
{
    // Завантаження зображення.
    cv::Mat image = cv::imread("C:\\Users\\oleg\\Desktop\\1.png");

    // Перетворення зображення до градацій сірого
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Бінаризація зображення для відокремлення люльок колеса
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY_INV);

    // Застосування медійного фільтра для видалення шуму
    cv::medianBlur(binary, binary, 11);

    // Знаходження контурів на зображенні
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Вибір найбільшого контуру, який має більше 5 точок
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

    // Обведення колеса огляду на зображенні
    cv::RotatedRect ellipse = cv::fitEllipse(wheelContour);
    cv::ellipse(image, ellipse, cv::Scalar(0, 255, 0), 2);

    // Показ зображення з обведеним колесом огляду
    cv::imshow("Wheel Detection", image);
    cv::waitKey(0);

    return 0;
}