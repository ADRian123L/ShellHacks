#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // Initialize k-NN algorithm
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    // Load images and preprocess them
    std::vector<cv::Mat> images_object_A, images_object_B;
    for (int i = 184; i <= 220; ++i) {
        std::string path = "images/object_A/IMG_0" + std::to_string(i) + ".JPG";
        cv::Mat     img  = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Could not read the image: " << path << std::endl;
            return -1;
        }
        cv::resize(img, img, cv::Size(64, 64));
        images_object_A.push_back(img);
    }

    for (int i = 116; i <= 183; ++i) {
        std::string path = "images/object_B/IMG_0" + std::to_string(i) + ".JPG";
        cv::Mat     img  = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Could not read the image: " << path << std::endl;
            return -1;
        }
        cv::resize(img, img, cv::Size(64, 64));
        images_object_B.push_back(img);
    }

    // Create training data and labels
    cv::Mat samples(0, 64 * 64, CV_32F), labels(0, 1, CV_32S);
    for (auto &img : images_object_A) {
        cv::Mat img_row = img.clone().reshape(1, 1);
        samples.push_back(img_row);
        labels.push_back(1);
    }

    for (auto &img : images_object_B) {
        cv::Mat img_row = img.clone().reshape(1, 1);
        samples.push_back(img_row);
        labels.push_back(0);
    }

    // Train k-NN model
    knn->train(samples, cv::ml::ROW_SAMPLE, labels);

    // Initialize video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;

        // Preprocess frame (resize and grayscale)
        cv::Mat resized_frame, gray_frame;
        cv::resize(frame, resized_frame, cv::Size(64, 64));
        cv::cvtColor(resized_frame, gray_frame, cv::COLOR_BGR2GRAY);

        // Prepare sample for k-NN prediction
        cv::Mat sample = gray_frame.reshape(1, 1);
        cv::Mat sample_32f;
        sample.convertTo(sample_32f, CV_32F);

        // Perform k-NN prediction
        cv::Mat response;
        knn->findNearest(sample_32f, 3, response);

        // Draw rectangle and label based on prediction
        if (response.at<float>(0, 0) == 1.0) {
            cv::rectangle(frame,
                          cv::Point(10, 10),
                          cv::Point(60, 60),
                          cv::Scalar(0, 255, 0),
                          2);
            cv::putText(frame,
                        "Object_A",
                        cv::Point(10, 80),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(0, 255, 0),
                        1);
        }
        else {
            cv::rectangle(frame,
                          cv::Point(10, 10),
                          cv::Point(60, 60),
                          cv::Scalar(0, 0, 255),
                          2);
            cv::putText(frame,
                        "Object_B",
                        cv::Point(10, 80),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(0, 0, 255),
                        1);
        }

        // Display the frame
        cv::imshow("Frame", frame);

        char c = (char) cv::waitKey(25);
        if (c == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
