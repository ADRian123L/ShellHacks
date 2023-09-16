/* g++ your_code_filename.cpp -o output_filename `pkg-config --cflags --libs
 * opencv4` */
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap(0); // open the webcam

    // Check if camera opened successfully
    if (!cap.isOpened()) {
        std::cout << "Error: Could not open camera." << std::endl;
        return 1;
    }

    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    // Load the trained model
    knn->read(cv::FileStorage("knn_model.xml").getRoot());

    // The dimensions should match the ones you used for training
    cv::Size imageSize(50, 50);

    while (true) {
        cv::Mat frame;
        cap >> frame; // get a new frame from the webcam

        // Convert the image to grayscale
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Resize the image
        cv::Mat resizedImage;
        cv::resize(gray, resizedImage, imageSize);

        // Flatten and convert to float
        cv::Mat testSample;
        resizedImage.reshape(1, 1).convertTo(testSample, CV_32F);

        cv::Mat response;
        knn->findNearest(testSample, 1, response);

        std::string label;
        if (response.at<float>(0, 0) == 0.0f) {
            label = "Object A";
        }
        else if (response.at<float>(0, 0) == 1.0f) {
            label = "Object B";
        }
        else {
            label = "Unknown";
        }

        // Draw the label on the frame
        cv::putText(frame,
                    label,
                    cv::Point(50, 50),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1,
                    cv::Scalar(0, 255, 0),
                    2);

        // Show the frame
        cv::imshow("Classification", frame);

        // Break the loop if 'q' is pressed
        if (cv::waitKey(30) >= 0) {
            break;
        }
    }

    return 0;
}
