#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

int main() {
    std::vector<cv::Mat> imagesForObjectA;
    std::vector<cv::Mat> imagesForObjectB;

    // Load and preprocess images for object A
    for (int i = 184; i <= 220; ++i) {
        std::string path = "images/object_A/IMG_0" + std::to_string(i) + ".jpg";
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cout << "Could not read the image at path: " << path << std::endl;
            return 1;
        }
        cv::resize(img, img, cv::Size(64, 64));
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        imagesForObjectA.push_back(img);
    }

    // Load and preprocess images for object B
    for (int i = 116; i <= 183; ++i) {
        std::string path = "images/object_B/IMG_0" + std::to_string(i) + ".jpg";
        cv::Mat img = cv::imread(path);
        if (img.empty()) {
            std::cout << "Could not read the image at path: " << path << std::endl;
            return 1;
        }
        cv::resize(img, img, cv::Size(64, 64));
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        imagesForObjectB.push_back(img);
    }

    // Initialize k-NN algorithm
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    // Prepare your data: flatten the image and put all in one big matrix
    int numSamples = 20;
    cv::Mat samples(numSamples * 2, 64 * 64, CV_32F);

    for (int i = 0; i < numSamples; ++i) {
        cv::Mat imgA = imagesForObjectA[i].clone().reshape(1, 1);
        imgA.convertTo(samples.row(i), CV_32F);

        cv::Mat imgB = imagesForObjectB[i].clone().reshape(1, 1);
        imgB.convertTo(samples.row(numSamples + i), CV_32F);
    }

    // Prepare labels
    cv::Mat labels;
    labels.push_back(cv::Mat::ones(cv::Size(1, numSamples), CV_32S));
    labels.push_back(cv::Mat::zeros(cv::Size(1, numSamples), CV_32S));

    // Train k-NN
    knn->train(samples, cv::ml::ROW_SAMPLE, labels);

    // At this point, the k-NN model is trained.
    // You can use knn->findNearest() to classify new samples in real-time.

    return 0;
}
