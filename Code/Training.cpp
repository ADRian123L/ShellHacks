/* g++ knn_training.cpp -o knn_training `pkg-config --cflags --libs opencv4` */
#include <opencv2/opencv.hpp>
#include <vector>

int main() {
    // Initialize k-NN algorithm
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();

    // Prepare your data: flatten the image and put all in one big matrix
    int numSamples = 20; // Assuming you have 20 samples for each object (you
                         // can adjust this number)
    cv::Mat samples(numSamples * 2,
                    256 * 256,
                    CV_32F); // Assuming your images are resized to 256x256

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

    // Now you can use knn->findNearest() to classify new samples
    // ...
    return 0;
}
