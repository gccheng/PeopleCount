#include <QtGui/QApplication>
#include "peoplecount.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/highgui/highgui.hpp>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    PeopleCount w;
    w.show();

    cv::Mat src_img;

#ifdef USE_VIDEO_CAPTURE
    cv::VideoCapture capture;
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    capture.open(0);
#else
    cv::Mat target_img = cv::imread("./left06.jpg");
#endif

#ifdef USE_GPU
    cv::gpu::GpuMat src_gpu, mono_gpu;
    cv::gpu::HOGDescriptor hog;
    hog.setSVMDetector(cv::gpu::HOGDescriptor::getDefaultPeopleDetector());
#else
    cv::HOGDescriptor hog;
    cv::Mat mono_img;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
#endif

    std::vector<cv::Rect> found;
    while(true) {
#ifdef USE_VIDEO_CAPTURE
        capture >> src_img;
#else
        src_img = target_img.clone();
#endif

#ifdef USE_GPU
        src_gpu.upload(src_img);
        cv::gpu::cvtColor(src_gpu, mono_gpu, CV_BGR2GRAY);
        hog.detectMultiScale(mono_gpu, found);
#else
        cv::cvtColor(src_img, mono_img, CV_BGR2GRAY);
        hog.detectMultiScale(mono_img, found);
#endif

        for(unsigned i = 0; i < found.size(); i++) {
            cv::Rect r = found[i];
            rectangle(src_img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
        }
        cv::imshow("test", src_img);

        int c = cv::waitKey(1);
        if( c == 27) break;
    }
    
    return a.exec();
}
