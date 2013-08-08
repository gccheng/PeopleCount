#include "digitrecognizer.h"

#include <time.h>

DigitRecognizer::DigitRecognizer(int ts, int cl, int sx, int sy, std::string path)
    : train_samples(ts), classes(cl), sizex(sx), sizey(sy), sizeimage(sx*sy), pathToImages(path)
{
    trainData = cv::Mat::zeros(classes*train_samples, sizeimage, CV_32FC1);
    trainClasses = cv::Mat::zeros(classes*train_samples, 1, CV_32FC1);
}

DigitRecognizer::~DigitRecognizer()
{
    delete this->pKNN;
}

int DigitRecognizer::test()
{
    cv::namedWindow("single", CV_WINDOW_AUTOSIZE);
    cv::namedWindow("all",CV_WINDOW_AUTOSIZE);

    learnFromImages();
    setClassifier();
    //runSelfTest(knearest);
    analyseImage("./images/buchstaben.png", true);

    return 0;
}

void DigitRecognizer::learnFromImages()
{
    cv::Mat img;
    char file[255];
    for (int i = 0; i < classes; i++)
    {
        for (int j=1; j<=2; j++) {
            sprintf(file, "%s/%d-%d.png", pathToImages.c_str(), i, j);
            img = cv::imread(file, 1);
            if (!img.data)
            {
                std::cout << "File " << file << " not found" << std::endl;
                exit(1);
            }
            cv::Mat outImage;
            preProcessImage(img, outImage);
            outImage.row(0).copyTo(trainData.row(i));
            //trainData.push_back(outImage.row(0));
            trainClasses.at<float>(i,0) = i;
        }
    }
}

void DigitRecognizer::runSelfTest()
{
    cv::Mat img;
    cv::Mat sample2 = cv::Mat::zeros(1, sizeimage, CV_32FC1);

    // SelfTest
    char file[255];
    int z = 0;
    while (z++ < 10)
    {
        int iSecret = rand() % 10;
        //cout << iSecret;
        sprintf(file, "%s/%d.png", pathToImages.c_str(), iSecret);
        img = cv::imread(file, 1);

        cv::Mat stagedImage;
        preProcessImage(img, stagedImage);
        stagedImage.row(0).copyTo(sample2.row(0));

        float detectedClass = pKNN->find_nearest(sample2, 1);
        if (iSecret != (int) ((detectedClass)))
        {
            std::cout << "Falsh. It is " << iSecret << " but guess is "
                 << (int) ((detectedClass));
            exit(1);
        }
        std::cout << "Right " << (int) ((detectedClass)) << std::endl;
        cv::imshow("single", img);
        cv::waitKey(0);
    }
}

std::string DigitRecognizer::analyseImage(std::string strImage, bool showWindow)
{
    cv::Mat image = cv::imread(strImage, 1);
    return analyseImage(image, showWindow);
}

std::string DigitRecognizer::analyseImage(cv::Mat& image, bool showWindow)
{
    cv::Mat sample2 = cv::Mat::zeros(1, sizeimage, CV_32FC1);
    cv::Mat gray, blur, thresh;
    cv::Mat digits = cv::Mat::zeros(image.size(), CV_8UC3);
    std::vector<std::vector<cv::Point> > contours;

    cv::cvtColor(image, gray, CV_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 1, 1);
    cv::adaptiveThreshold(blur, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 2);
    cv::findContours(thresh, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::ostringstream ossRet;
    for (size_t i = 0; i < contours.size(); i++)
    {
        std::vector<cv::Point> cnt = contours[i];
        if (cv::contourArea(cnt) > 5)  ///50
        {
            cv::Rect rec = cv::boundingRect(cnt);
            if (rec.height > 8)  ///28
            {
                cv::Mat roi = image(rec);
                cv::Mat stagedImage;
                preProcessImage(roi, stagedImage);

                stagedImage.row(0).copyTo(sample2.row(0));
                float result = pKNN->find_nearest(sample2, 1);
                /*cv::rectangle(image, cv::Point(rec.x, rec.y),
                          cv::Point(rec.x + rec.width, rec.y + rec.height),
                          cv::Scalar(0, 0, 255), 1);*/
                ossRet << result;
                if (showWindow) {
                    std::ostringstream oss;
                    oss << result;
                    cv::putText(digits, oss.str(), cv::Point(rec.x, rec.y+rec.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1, 8, false);
                }
            }
        }
    }

    if (showWindow) {
        cv::imshow("Digits", digits);
        cv::imshow("Image", image);
        cv::waitKey(5);
    }
    return ossRet.str();
}

std::string DigitRecognizer::analyseLocationImage(cv::Mat& image, bool showWindow)
{
    cv::Mat showImage = image.clone();
    cv::Mat sample2 = cv::Mat::zeros(1, sizeimage, CV_32FC1);
    cv::Mat digits = cv::Mat::zeros(image.size()*2, CV_8UC3);

    cv::Point base(7,3);
    int xoffset = 7;
    int yoffset = 7;
    std::ostringstream ossRet;
    cv::Rect rects[] = {cv::Rect(0*xoffset,0,8,9)+base,
                        cv::Rect(1*xoffset,0,8,9)+base,
                        cv::Rect(3*xoffset+1,0,8,9)+base,
                        cv::Rect(4*xoffset+1,0,8,9)+base,
                        cv::Rect(6*xoffset+1,0,8,9)+base,
                        cv::Rect(7*xoffset+1,0,8,9)+base,
                        cv::Rect(8*xoffset+1,0,8,9)+base,
                        cv::Rect(9*xoffset+1,0,8,9)+base,
                        cv::Rect(0*xoffset,yoffset+7,8,9)+base,
                        cv::Rect(1*xoffset,yoffset+7,8,9)+base,
                        cv::Rect(3*xoffset+1,yoffset+7,8,9)+base,
                        cv::Rect(4*xoffset+1,yoffset+7,8,9)+base,
                        cv::Rect(6*xoffset+1,yoffset+7,8,9)+base,
                        cv::Rect(7*xoffset+1,yoffset+7,8,9)+base,
                        cv::Rect(8*xoffset+1,yoffset+7,8,9)+base,
                        cv::Rect(9*xoffset+1,yoffset+7,8,9)+base,
                       };

    for (size_t i = 0; i < 16; i++)
    {
        cv::Rect rec = rects[i];
        cv::Mat roi = image(rec);
        cv::Mat stagedImage;

        preProcessImage(roi, stagedImage);

        stagedImage.row(0).copyTo(sample2.row(0));
        float result = pKNN->find_nearest(sample2, 1);
        /*cv::rectangle(showImage, cv::Point(rec.x, rec.y),
                      cv::Point(rec.x + rec.width, rec.y + rec.height),
                      cv::Scalar(0, 0, 255), 1);*/
        ossRet << result;
        if (showWindow) {
            std::ostringstream oss;
            oss << result;
            cv::putText(digits, oss.str(), cv::Point(rec.x*2, 2*(rec.y+rec.height)), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 1, 8, false);
        }
    }

    if (showWindow) {
        cv::imshow("Digits", digits);
        cv::imshow("Image", showImage);
        cv::waitKey(5);
    }
    return ossRet.str();
}

void DigitRecognizer::setClassifier()
{
    pKNN = new cv::KNearest(trainData, trainClasses);
}


void DigitRecognizer::preProcessImage(const cv::Mat& inImage, cv::Mat& outImage)
{
    cv::Mat grayImage,blurredImage,thresholdImage,contourImage,regionOfInterest;
    std::vector<std::vector<cv::Point> > contours;

    if (inImage.channels()==3) {
        cv::cvtColor(inImage,grayImage , CV_BGR2GRAY);
    } else {
        inImage.copyTo(grayImage);
    }
    blurredImage = grayImage;
    //cv::GaussianBlur(grayImage, blurredImage, cv::Size(3, 3), 2, 2);
    cv::adaptiveThreshold(blurredImage, thresholdImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 3, 0);

    int rows = thresholdImage.rows;
    int cols = thresholdImage.cols;
    for (int r=0; r<rows; r++) {
        for (int c=0; c<cols; c++) {
            uchar p = thresholdImage.at<uchar>(r,c);
            if (p==0) continue;
            bool allZeros = true;

            for (int i=-1; i<=1; i++) {
                for (int j=-1; j<=1; j++) {
                    if (i==0 && j==0) continue;
                    if (allZeros && (r+i>-1) && (r+i<rows) && (c+j>-1) && (c+j<cols)
                            && (thresholdImage.at<uchar>(r+i,c+j)==p)) {
                        allZeros = false;
                    }
                }
            }
            if (allZeros == true) {
                thresholdImage.at<uchar>(r,c) = 0;
            }
        }
    }

    thresholdImage.copyTo(contourImage);
    cv::findContours(contourImage, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    int idx = 0;
    size_t area = 0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        if (area < contours[i].size() )
        {
            idx = i;
            area = contours[i].size();
        }
    }

    cv::Rect rec = cv::boundingRect(contours[idx]);

    regionOfInterest = thresholdImage(rec);
    cv::resize(regionOfInterest,outImage, cv::Size(sizex, sizey));
    outImage = outImage.reshape(0, sizeimage).t();
}
