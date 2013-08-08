#ifndef HISTOGRAMS_H
#define HISTOGRAMS_H

#include <opencv2/core/core.hpp>
#include <vector>


class Utility {

public:
    // build integral histogram of HoG
    static cv::Mat HogComp(const cv::Mat& img)
    {
        int width = img.cols;
        int height = img.rows;
        int nBins = 8;
        int angleBase = 360/nBins;
        cv::Mat xComp(cv::Size(width, height), CV_32FC1);
        cv::Mat yComp(cv::Size(width, height), CV_32FC1);

        cv::Sobel(img, xComp, CV_32F, 1, 0);
        cv::Sobel(img, yComp, CV_32F, 0, 1);

        //cv::Mat hist(1, nBins, CV_32FC1);
        cv::Mat hist = cv::Mat::zeros(1, nBins, CV_32FC1);

        for (int y=0; y<height; y++) {
            const float *xcompRow = xComp.ptr<float>(y);
            const float *ycompRow = yComp.ptr<float>(y);

            for (int x=0; x<width; x++) {
                float shiftX = xcompRow[x];
                float shiftY = ycompRow[x];
                float magnitude0 = sqrt(shiftX*shiftX + shiftY*shiftY);
                float magnitude1 = magnitude0;
                int bin0, bin1;

                float orientation = cv::fastAtan2(shiftY, shiftX);

                // split the magnitude to two adjacent bins
                float fbin = orientation/angleBase;
                bin0 = cvFloor(fbin) % nBins;
                bin1 = (bin0+1)%nBins;
                float weight0 = 1 - (fbin-bin0);
                float weight1 = 1 - weight0;

                magnitude0 *= weight0;
                magnitude1 *= weight1;

                hist.at<float>(bin0) += magnitude0;
                hist.at<float>(bin1) += magnitude1;
            }
        }

        return hist;
    }

    // build integral histogram of HoF
   static cv::Mat HofComp(const cv::Mat& flow)     // input flow image (2 channels)
    {
        int nBins = 8;
        float angleBase = 360/nBins;
        int width = flow.cols;
        int height = flow.rows;
        cv::Mat xComp(cv::Size(width, height), CV_32FC1);
        cv::Mat yComp(cv::Size(width, height), CV_32FC1);

        //cv::Mat hist(1, nBins+1, CV_32FC1);
        cv::Mat hist = cv::Mat::zeros(1, nBins+1, CV_32FC1);

        if (flow.channels()==2)
        {
            std::vector<cv::Mat> xyChannel;
            cv::split(flow, xyChannel);
            xComp = xyChannel[0];
            yComp = xyChannel[1];

            for (int y=0; y<height; y++) {
                const float *xcompRow = xComp.ptr<float>(y);
                const float *ycompRow = yComp.ptr<float>(y);

                for (int x=0; x<width; x++) {
                    float shiftX = xcompRow[x];
                    float shiftY = ycompRow[x];
                    float magnitude0 = sqrt(shiftX*shiftX + shiftY*shiftY);
                    float magnitude1 = magnitude0;
                    int bin0, bin1;

                    if (magnitude0 <= 1.0)
                    {
                        bin0 = nBins;  // the zero bin is the last one
                        magnitude0 = 1.0;
                        bin1 = 0;
                        magnitude1 = 0;
                    }
                    else
                    {
                        float orientation = cv::fastAtan2(shiftY, shiftX);
                        //std::cout << orientation << std::endl;

                        // split the magnitude to two adjacent bins
                        float fbin = orientation/angleBase;
                        bin0 = cvFloor(fbin) % nBins;
                        bin1 = (bin0+1)%nBins;
                        float weight0 = 1 - (fbin-bin0);
                        float weight1 = 1 - weight0;

                        magnitude0 *= weight0;
                        magnitude1 *= weight1;
                    }

                    hist.at<float>(bin0) += magnitude0;
                    hist.at<float>(bin1) += magnitude1;
                }
            }
        }

        return hist;
    }

};


#endif // HISTOGRAMS_H

