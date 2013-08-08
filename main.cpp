#include <QtGui/QApplication>
#include "peoplecount.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <functional>

#include "histograms.h"
#include "digitrecognizer.h"
#include "counter.h"

// Trackbar callback
int trackvalue = 5;             // position of track  bar
unsigned char lut[256];         // gamma function
void speedCallBack(int pos, void *data)
{
    Counter *pc = (Counter*)data;
    pc->updateTrackValue(pos);
}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    PeopleCount w;
    w.hide();


    Counter c;
    w.connect(&c, SIGNAL(sendImage(QImage)), &w, SLOT(showImage(QImage)), Qt::QueuedConnection);
    // Windows
    cv::namedWindow("Passenger Count");
    cv::createTrackbar("Speed", "Passenger Count", &trackvalue, 10, speedCallBack, &c);
    c.startCount("../PeopleCount/Front_Door.mp4", 'F');

    // for gamma correction
    for (int i=0; i<256; ++i)    {
        lut[i] = std::pow((float)(i/255.0), 0.8) * 255.0;
    }

    //DigitRecognizer dr(1, 10, 20, 30);
    //dr.test();

    return a.exec();
}
