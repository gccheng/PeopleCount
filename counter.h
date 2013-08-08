#ifndef COUNTER_H
#define COUNTER_H

#include <QObject>
#include <QImage>

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

#include "digitrecognizer.h"

class Counter : public QObject
{
    Q_OBJECT
public:
    Counter(const std::string& windowName = "Passenger Count");

public:
    int startCount(std::string file, char frontRear = 'F');
    void setFrontDoor();
    void setRearDoor();
    void setPerspectiveTransform(const std::vector<cv::Point2f>& src, const std::vector<cv::Point2f>& dst);
    void updateTrackValue(int value);

signals:
    void sendImage(const QImage &img);

private:
    void addPoints(std::list<std::vector<cv::Point2f> >& tracks, std::vector<cv::Point2f> points, float frame);
    void drawtrajectory(std::vector<cv::Point2f>& track, cv::Mat& image);
    void gammaCorrection(cv::Mat& image);
    void generateTrackingPoints(std::vector<cv::Point2f> &trackPoints);
    int  getMajorityDirection(const std::list<std::vector<cv::Point2f> >& trajectories, int time);
    void getStartTimes(const std::list<std::vector<cv::Point2f> >& trajectories, std::vector<int>& startTimes, int fno);
    bool isValidTrack(const std::vector<cv::Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length);
    void showResultImage(cv::Mat &img, int onPassengers, int offPassengers, std::string gps="", std::string speed="");
    int  onOroff(const std::vector<cv::Point2f>& track);
    int  onOrOffStartEnd(const std::vector<cv::Point2f>& track);
    int  onOrOffHistOrient(const std::vector<cv::Point2f>& track);
    int  updateModel(std::vector<std::list<int> >& models, const std::vector<cv::Point2f>& track, int onOrOff);
    std::vector<cv::Point2f> lastPoints(std::list<std::vector<cv::Point2f> >& tracks);

private:
    // tracking & clustering parameters
    int t_min;                  // if two trajectories are less than t_min apart, their distance is 0
    int t_max;                  // if two trajectories are more than t_max apart, their distance is 1
    int track_len;              // minimum length of a valid trajectory
    int min_track_group_off;    // minimum number of trajectories for each alighting passenger
    int min_track_group_on;     // minimum number of trajectories for each boarding passenger
    float t_threshold_off;      // threshold for alighting trajectories between (t_min, t_max)
    float t_threshold_on;       // threshold for boarding trajectories between (t_min, t_max)

    // predefined tripwire and door region
    cv::Point2f startTripWire;  // starting point of tripwire
    cv::Point2f endTripWire;    // ending point of tripwire
    cv::Rect rectDoor;          // bounding box for background computation on the door (front/rear)
    float baseline_orient;      // the orientation (in degrees) of the tripwire
    int voffset;                // vertical offset of the tripwire
    int hoffset;                // horizontal offset of the tripwire

    // gui related
    std::string windowName;     // name of the qt-opencv window to show the result
    int trackvalue;             // position of track  bar
    float speedratio;           // play speed

    // image & perspective transformation
    unsigned char lut[256];     // gamma function
    cv::Mat perspectiveM;       // perspective transform

    // video info
    int fps;                    //
};

#endif // COUNTER_H
