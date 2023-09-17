#include <iostream>
#include <opencv2/opencv.hpp>

void detectAndDrawBox(cv::Mat          &frame,
                      const cv::Scalar &lower,
                      const cv::Scalar &upper,
                      const cv::Scalar &color,
                      bool             &flag,
                      int              &counter);

int main() {

    cv::VideoCapture cap(0); // Open the webcam

    if (!cap.isOpened()) {
        std::cout << "ERROR: Could not open camera" << std::endl;
        return EXIT_FAILURE;
    }

    bool blueFlag     = false;
    bool greenFlag    = false;
    int  blueCounter  = 0;
    int  greenCounter = 0;

    while (true) {
        cv::Mat frame;
        cap >> frame; // Capture a frame from webcam

        if (frame.empty()) {
            std::cerr << "Received empty frame" << std::endl;
            break;
        }

        detectAndDrawBox(
            frame,
            cv::Scalar(100, 50, 50),
            cv::Scalar(140, 255, 255),
            cv::Scalar(237, 114, 47), // Changed to BGR (237, 114, 47)
            blueFlag,
            blueCounter);

        detectAndDrawBox(
            frame,
            cv::Scalar(35, 50, 50),
            cv::Scalar(85, 255, 255),
            cv::Scalar(88, 205, 124), // Changed to BGR (88, 205, 124)
            greenFlag,
            greenCounter);

        cv::imshow("Color Detection", frame);

        if (cv::waitKey(1) >= 0) {
            break;
        }
    }

    return EXIT_SUCCESS;
}

void detectAndDrawBox(cv::Mat          &frame,
                      const cv::Scalar &lower,
                      const cv::Scalar &upper,
                      const cv::Scalar &color,
                      bool             &flag,
                      int              &counter) {

    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv, lower, upper, mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(
        mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Rect biggestBox;
    double   maxArea = 0;
    for (const auto &contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea    = area;
            biggestBox = cv::boundingRect(contour);
        }
    }

    if (maxArea > 700) {
        cv::rectangle(frame, biggestBox, cv::Scalar(9, 9, 255), 2);
        std::string text =
            (color[2] == 47)
                ? "Blue Object"
                : "Green Object"; // Based on the B value of blue and green

        if (!flag) {
            counter++;
            flag = true;
        }

        cv::putText(frame,
                    text + " Count: " + std::to_string(counter),
                    cv::Point(biggestBox.x, biggestBox.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2);
    }
    else {
        flag = false;
    }
}
