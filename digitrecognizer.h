#ifndef DIGITRECOGNIZER_H
#define DIGITRECOGNIZER_H

#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <stdio.h>

class DigitRecognizer
{
public:
    DigitRecognizer(int ts = 1, int cl = 10, int sx = 20, int sy = 30, std::string path = "./images");
    ~DigitRecognizer();

public:
    void learnFromImages();
    void runSelfTest();
    std::string analyseImage(std::string strImage, bool showWindow = true);
    std::string analyseImage(cv::Mat& image, bool showWindow = true);
    std::string analyseLocationImage(cv::Mat& image, bool showWindow = true);
    void setClassifier();
    int test();

private:
    void preProcessImage (const cv::Mat& inImage, cv::Mat& outImage);

protected:
    int train_samples;            // # training samples of each class
    int classes;                  // # classes
    int sizex;                    // x-length of normalized image
    int sizey;                    // y-length of normalized image
    int sizeimage;                // size of normalized image
    std::string pathToImages;     // path to training images

    cv::Mat trainData;             // normalized training data for kNN or other classifier
    cv::Mat trainClasses;          // class labels of the training data

    cv::KNearest *pKNN;
};

#endif // DIGITRECOGNIZER_H
