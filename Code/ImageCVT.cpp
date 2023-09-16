/* To run the code: g++ your_code_file.cpp -o output_program `pkg-config
 * --cflags --libs opencv4` */
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

int main() {
    // Create vectors to hold the images for each class
    std::vector<cv::Mat> images_object_A;
    std::vector<cv::Mat> images_object_B;

    // Assume we have 10 images for each class
    // Load and preprocess images for object A
    for (int i = 1; i <= 10; i++) {
        std::string path = "images/object_A/image" + std::to_string(i) + ".jpg";
        cv::Mat     img  = cv::imread(path);

        // Check if image was loaded successfully
        if (img.empty()) {
            std::cout << "Could not read the image at path: " << path
                      << std::endl;
            return 1;
        }

        // Resize the image to 64x64 pixels
        cv::resize(img, img, cv::Size(64, 64));

        // Convert the image to grayscale (optional)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

        // Add the image to the list
        images_object_A.push_back(img);
    }

    // Load and preprocess images for object B
    for (int i = 1; i <= 10; i++) {
        std::string path = "images/object_B/image" + std::to_string(i) + ".jpg";
        cv::Mat     img  = cv::imread(path);

        // Check if image was loaded successfully
        if (img.empty()) {
            std::cout << "Could not read the image at path: " << path
                      << std::endl;
            return 1;
        }

        // Resize the image to 64x64 pixels
        cv::resize(img, img, cv::Size(64, 64));

        // Convert the image to grayscale (optional)
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

        // Add the image to the list
        images_object_B.push_back(img);
    }

    std::cout << "Loaded and preprocessed all images." << std::endl;
    return 0;
}
