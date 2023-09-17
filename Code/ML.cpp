#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // Initialize SVM algorithm
    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);

    // Load images and preprocess them
    std::vector<cv::Mat> images_object_A, images_object_B;
    for (int i = 645; i <= 948; ++i) {
        std::string path = "images/object_A/IMG_0" + std::to_string(i) + ".JPG";
        cv::Mat     img  = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Could not read the image: " << path << std::endl;
            return -1;
        }
        cv::resize(img, img, cv::Size(64, 64));
        images_object_A.push_back(img);
    }

    for (int i = 949; i <= 1206; ++i) {
        std::string path;
        if (i < 1000) {
            path = "images/object_B/IMG_0" + std::to_string(i) + ".JPG";
        }
        else if (i == 1000) {
            continue;
        }
        else {

            path = "images/object_B/IMG_" + std::to_string(i) + ".JPG";
        }
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Could not read the image: " << path << std::endl;
            return -1;
        }
        cv::resize(img, img, cv::Size(64, 64));
        images_object_B.push_back(img);
    }

    // Create training data and labels
    cv::Mat samples(0, 64 * 64, CV_32F), labels;
    for (const auto &img : images_object_A) {
        cv::Mat img_row = img.reshape(1, 1);
        img_row.convertTo(img_row, CV_32F);
        samples.push_back(img_row);
        labels.push_back(1);
    }

    for (const auto &img : images_object_B) {
        cv::Mat img_row = img.reshape(1, 1);
        img_row.convertTo(img_row, CV_32F);
        samples.push_back(img_row);
        labels.push_back(0);
    }

    // Train SVM model
    cv::Ptr<cv::ml::TrainData> td =
        cv::ml::TrainData::create(samples, cv::ml::ROW_SAMPLE, labels);
    svm->train(td);

    // Initialize video capture
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Error opening video stream" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Preprocess frame (resize and convert to grayscale)
        cv::Mat gray_frame;
        cv::resize(frame, gray_frame, cv::Size(64, 64));
        cv::cvtColor(gray_frame, gray_frame, cv::COLOR_BGR2GRAY);

        // Prepare the sample and predict
        cv::Mat sample = gray_frame.reshape(1, 1);
        sample.convertTo(sample, CV_32F);
        float response = svm->predict(sample);

        // Display prediction on frame
        std::string label = response == 1.0 ? "Object_A" : "Object_B";
        cv::putText(frame,
                    label,
                    cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX,
                    5,
                    cv::Scalar(0, 255, 0),
                    8);

        // Show the frame
        cv::imshow("SVM Object Classification", frame);

        // Break the loop if 'ESC' is pressed
        char c = (char) cv::waitKey(25);
        if (c == 27) {
            break;
        }
    }

    // Release the video capture
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
