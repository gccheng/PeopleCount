#include "counter.h"

#include <numeric>
#include <math.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "configure.h"
#include "histograms.h"

Counter::Counter(const std::string& window/* = "Passenger Count"*/)
{
    // Time & Track  3, 24
    t_min = 3;                  // if two trajectories are less than t_min apart, their distance is 0  3 rear
    t_max = 24;                 // if two trajectories are more than t_max apart, their distance is 1  24 rear
    baseline_orient = 27;       // the orientation (in degrees) of the tripwire
    track_len = 12;             // minimum length of a valid trajectory
    t_threshold_off = 0.7;      // threshold for alighting trajectories between (t_min, t_max)  0.6 rear
    t_threshold_on  = 1.0;      // threshold for boarding trajectories between (t_min, t_max)  0.6 rear

    // Board 20  Alight 50
    min_track_group_off = 50;   // minimum number of trajectories for each alighting passenger //50 for rear door
    min_track_group_on = 50;    // minimum number of trajectories for each boarding passenger //35 for rear door

    // Tripwire -20 -10
    voffset = -30;              // vertical offset of the tripwire
    hoffset = -20;              // horizontal offset of the tripwire

    trackvalue = 5;             // position of track  bar
    speedratio = trackvalue>=5 ? trackvalue-4 : 1.0/(6-trackvalue);   // play speed
    windowName = window;

    // for gamma correction
    for (int i=0; i<256; ++i)    {
        lut[i] = std::pow((float)(i/255.0), 0.8) * 255.0;
    }

    // perspective transform
    std::vector<cv::Point2f> src(0), dst(0);
    setPerspectiveTransform(src, dst);
}

int Counter::startCount(std::string file, char frontRear/* = 'F'*/)
{
    cv::VideoCapture cap(file.c_str());
    if (!cap.isOpened()) {
        std::cout << "Could not open file" << std::endl;
        return 1;
    }
    fps = 1000/cap.get(CV_CAP_PROP_FPS);
    //int frate = 1000/fps;
    int frate = 20;
    int dumy = 13700;  // @debug  13700  15840   18246   18890   21900

    // Location recognition
    DigitRecognizer dr(1,10,5,7, "./origImages");
    dr.learnFromImages();
    dr.setClassifier();

    // set parameters
    if ('F'==frontRear) {
        setFrontDoor();
    } else {
        setRearDoor();
    }

    std::vector<cv::Point2f> tripWire;                  // points on the tripwire
    std::list<std::vector<cv::Point2f> > trajectories;  // a list of trajectories being tracked
    std::vector<std::list<int> > on_models;             // each model is a list of start times
    std::vector<std::list<int> > off_models;
    float mean_x=0.0f, mean_y=0.0f, var_x=0.0f, var_y=0.0f, length=0.0f;    // trajectory stats

    cv::Mat capframe, frame, image, gray, prevGray, location;
    cv::Mat doorHistBG, door, doorHist;
    cv::Size winSize(31,31);            // window size for optical flow computation
    cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    int onPassengers = 0;
    int offPassengers = 0;
    int missPassengers = 0;
    int histSize = 2;                   // size of background histogram
    float range[] = { 0, 256 };         // range of pixel values for histogram calculation
    const float* histRange = { range }; //
    std::string prevGPS, currGPS, speed;

    generateTrackingPoints(tripWire);

    while (true) {
        int fno = cap.get(CV_CAP_PROP_POS_FRAMES);
        if (fno>=dumy) {
            std::cout << "";
        }

        cap >> capframe;
        if (capframe.empty()) break;

        frame = capframe(cv::Rect(0,0,580,450));
        //cv::warpPerspective(frame, frame, M, frame.size() );
        frame.copyTo(image);
        cv::cvtColor(image, gray, CV_BGR2GRAY);
        //gammaCorrection(gray);  // note: it becomes worse with Gamma (anti-) correction

        if (prevGray.empty()) {
            gray.copyTo(prevGray);
        }

        // check gps location
        location = capframe(cv::Rect(810, 90, 90, 30));
        currGPS = dr.analyseLocationImage(location, false);
        /*int gpsDistance = 0;
        if (!prevGPS.empty()) {
            std::inner_product(prevGPS.begin(), prevGPS.end(), currGPS.begin(), gpsDistance,
                               std::plus<int>(), std::not_equal_to<char>());
        }
        // add points to trajectories /// NEED TO KNOW THAT GPS DOESN'T CHANGE FOR SEVERAL FRAMES
        if(trajectories.size()<tripWire.size()-10 && gpsDistance<3) { //160 0.8
            addPoints(trajectories, tripWire, fno);
        }*/

        // check if door is closed
        door = gray(rectDoor);
        if (fno<5) {
            cv::Mat tmpDoorHistBG;
            //cv::calcHist(&door, 1, 0, cv::Mat(), tmpDoorHistBG, 1, &histSize, &histRange, true, false);
            tmpDoorHistBG = Utility::HogComp(door);
            cv::normalize(tmpDoorHistBG, tmpDoorHistBG, 1, 0, cv::NORM_L2, -1, cv::Mat());
            if (doorHistBG.empty()) {
                doorHistBG = tmpDoorHistBG;
            } else {
                cv::addWeighted(doorHistBG, 0.7, tmpDoorHistBG, 0.3, 0, doorHistBG, -1);
            }
        }
        //cv::calcHist(&door, 1, 0, cv::Mat(), doorHist, 1, &histSize, &histRange, true, false);
        doorHist = Utility::HogComp(door);
        cv::normalize(doorHist, doorHist, 1, 0, cv::NORM_L2, -1, cv::Mat());
        //float similarityDoor = doorHistBG.dot(doorHist);
        float similarityDoor = cv::compareHist(doorHistBG, doorHist, CV_COMP_CORREL);
        bool bDoorOpen = similarityDoor<0.9;

        // add points to trajectories
        if(trajectories.size()<tripWire.size()-10 && bDoorOpen) { //160 0.8
            addPoints(trajectories, tripWire, fno);
        }

        std::vector<uchar> status;
        std::vector<float> err;
        std::vector<cv::Point2f> nextPoints;
        std::vector<cv::Point2f> prevPoints = lastPoints(trajectories);
        if (prevPoints.empty()==false) {
            cv::calcOpticalFlowPyrLK(prevGray, gray, prevPoints, nextPoints, status, err, winSize, 3, termcrit, 0, 0.001);
        }

        int i=0;
        std::list<std::vector<cv::Point2f> >::iterator iTrack = trajectories.begin();
        for (; iTrack!=trajectories.end(); i++) {
            int szTrack = iTrack->size();
            isValidTrack(*iTrack, mean_x, mean_y, var_x, var_y, length);

            if ((szTrack>3) && (var_x<1.0f) && (var_y<1.0f)) { // stationary points
                iTrack = trajectories.erase(iTrack);
            } else if ((!status[i] || err[i]>13.0) && (szTrack>10)) { // lost of tracking
                iTrack->at(0).y = 1.0;
                iTrack++;
            } else if (szTrack>80) { // too long, remove  120
                iTrack = trajectories.erase(iTrack);
            } else if (szTrack>30) { // long trajectory, try to check 80
                iTrack->at(0).y = 2.0;
                iTrack->push_back(nextPoints[i]);
                iTrack++;
            } else {
                iTrack->push_back(nextPoints[i]);
                iTrack++;
            }
        }

        // update models according to the direction of trajectories
        std::vector<int> startTimes;
        getStartTimes(trajectories, startTimes, fno);
        std::vector<int>::iterator iTime = startTimes.begin();
        for (; iTime!=startTimes.end(); iTime++) {
            int overall_direction = getMajorityDirection(trajectories, *iTime);
            for (i=0, iTrack=trajectories.begin(); iTrack!=trajectories.end(); i++) {
                drawtrajectory(*iTrack, image);
                if (((int)(iTrack->at(0).x) == *iTime) && (iTrack->at(0).y>0.0f)) { // only use trajectories long enough
                    bool validTrack = isValidTrack(*iTrack, mean_x, mean_y, var_x, var_y, length);
                    int onoff = onOroff(*iTrack);
                    if (validTrack && (onoff==overall_direction)) {
                        switch(onoff) {
                        case 0: {offPassengers = updateModel(off_models, *iTrack, onoff);
                            /*std::vector<cv::Point2f>::iterator iit = iTrack->begin();
                            while (iit!=iTrack->end()) {
                                std::cout << iit->x << " " << iit->y << " ";
                                ++iit;
                            }
                            std::cout << std::endl;*/
                            iTrack = trajectories.erase(iTrack);
                            continue;}
                        case 1: {onPassengers = updateModel(on_models, *iTrack, onoff);
                            iTrack = trajectories.erase(iTrack);
                            continue;}
                        case 2: {missPassengers++;
                            iTrack = trajectories.erase(iTrack);
                            continue;}
                        default: std::cout << "Error: Wrong branch!" << std::endl;
                        }
                    }
                    if ((int)(iTrack->at(0).y) == 1) { // lost tracking
                        iTrack = trajectories.erase(iTrack);
                    }
                }

                iTrack++;
            }
        }

        //cv::rectangle(image, rectDoor, cv::Scalar(0,255,0));
        showResultImage(image, onPassengers, offPassengers, currGPS, speed);

        if ((char)cv::waitKey(frate/speedratio)==27) break;
        cv::swap(prevGray, gray);
        std::swap(currGPS, prevGPS);
    }

    return 0;
}

void Counter::setFrontDoor()
{
    //rectDoor = cv::Rect(180, 20, 40, 20);
    rectDoor = cv::Rect(160, 20, 80, 20);     // for HOG use
    startTripWire = cv::Point2f(252+hoffset, 217+voffset);
    endTripWire = cv::Point2f(377+hoffset, 153+voffset);
    baseline_orient = cv::fastAtan2(startTripWire.y-endTripWire.y, endTripWire.x-startTripWire.x);

    t_min = 3;
    t_max = 24;

    t_threshold_off = 0.6;
    t_threshold_on  = 1.0;
}

void Counter::setRearDoor()
{
    //rectDoor = cv::Rect (70, 90, 25, 20);
    rectDoor = cv::Rect (40, 70, 85, 30);     // for HOG use
    startTripWire = cv::Point2f(96+hoffset, 336+voffset);
    endTripWire = cv::Point2f(207+hoffset, 289+voffset);
    baseline_orient = cv::fastAtan2(startTripWire.y-endTripWire.y, endTripWire.x-startTripWire.x);

    t_min = 3;
    t_max = 24;

    t_threshold_off = 0.6;
    t_threshold_on  = 1.0;
}

void Counter::setPerspectiveTransform(const std::vector<cv::Point2f>& src,
                                      const std::vector<cv::Point2f>& dst)
{
    if (src.empty() || dst.empty()) {
        perspectiveM = cv::Mat::eye(3, 3, CV_32FC1);
    } else {
        perspectiveM = cv::getPerspectiveTransform(src, dst);
    }
}

struct RetrieveKey {
    int operator()(const std::pair<int,int>& keyValuePair) const {
        return keyValuePair.first;
    }
};

void Counter::getStartTimes(const std::list<std::vector<cv::Point2f> >& trajectories, std::vector<int>& startTimes, int fno)
{
    std::map<int, int> startMap;
    std::list<std::vector<cv::Point2f> >::const_iterator iTrack = trajectories.begin();
    for (; iTrack!=trajectories.end(); iTrack++) {
        if ((int)(fno-iTrack->size())>track_len) {
            startMap[iTrack->at(0).x]++;
        }
    }

    std::transform(startMap.begin(), startMap.end(), std::back_inserter(startTimes), RetrieveKey());
}

void Counter::gammaCorrection(cv::Mat& image)
{
    cv::MatIterator_<uchar> it, end;
    for (it=image.begin<uchar>(), end=image.end<uchar>(); it!=end; it++)
    {
        *it = lut[(*it)];
    }
}

void Counter::generateTrackingPoints(std::vector<cv::Point2f> &trackPoints)
{
    float slope = (float)(endTripWire.y-startTripWire.y)/(endTripWire.x-startTripWire.x);
    int shrinkRight = 30;       // remove some points on the right
    int shrinkLeft = 20;        // remove some points on the left
    for (float x=startTripWire.x+shrinkLeft, inc=1.0f; x<endTripWire.x-shrinkRight; x+=0.5, inc+=0.5) {
        cv::Point2f pt(x, startTripWire.y+(inc+shrinkLeft)*slope);
        trackPoints.push_back(pt);

        cv::Point2f time(0, 0);
        cv::vector<cv::Point2f> vec;
        vec.push_back(time);
        vec.push_back(pt);
    }
}

std::vector<cv::Point2f> Counter::lastPoints(std::list<std::vector<cv::Point2f> >& tracks)
{
    std::vector<cv::Point2f> points;
    std::list<std::vector<cv::Point2f> >::iterator it = tracks.begin();
    for (; it!=tracks.end(); ++it)
    {
        points.push_back(it->back());
    }

    return points;
}

void Counter::updateTrackValue(int value)
{
    trackvalue = value;
    speedratio = trackvalue>=5 ? trackvalue-4 : 1.0/(6-trackvalue);
}


void Counter::addPoints(std::list<std::vector<cv::Point2f> >& tracks, std::vector<cv::Point2f> points, float frame)
{
    std::vector<cv::Point2f>::iterator it= points.begin();
    for (; it!=points.end(); it++)
    {
        std::vector<cv::Point2f> vec;
        vec.push_back(cv::Point2f(frame, 0));
        vec.push_back(*it);
        tracks.push_back(vec);
    }
}

// draw trajectory {desc}, which is in {level}th pyramid, in {image}
void Counter::drawtrajectory(std::vector<cv::Point2f>& track, cv::Mat& image)
{
    std::vector<cv::Point2f>::iterator it = track.begin();
    float length = track.size();
    it++;  // skip time element at the beginning
    cv::Point2f point0 = *it;

    float j = 0;
    for(it++; it!=track.end() ; ++it, ++j)
    {
        cv::Point2f point1(*it);

        cv::line(image, point0, point1, cv::Scalar(0, cv::saturate_cast<uchar>(255.0*(j+1.0)/length), 0), 2, 8, 0);
        point0 = point1;
    }
    cv::circle(image, point0, 1, cv::Scalar(255, 0, 0), -1, 8, 0);
}

// check if a track is valid
bool Counter::isValidTrack(const std::vector<cv::Point2f>& track, float& mean_x, float& mean_y, float& var_x, float& var_y, float& length)
{
    mean_x = mean_y = var_x = var_y = length = 0.0f;

    int size = track.size();
    for(int i = 1; i < size; i++) {
        mean_x += track[i].x;
        mean_y += track[i].y;
    }
    mean_x /= size-1;
    mean_y /= size-1;

    for(int i = 1; i < size; i++) {
        var_x += (track[i].x-mean_x)*(track[i].x-mean_x);
        var_y += (track[i].y-mean_y)*(track[i].y-mean_y);
    }
    var_x /= size-1;
    var_y /= size-1;
    var_x = sqrt(var_x);
    var_y = sqrt(var_y);
    // remove static trajectory
    if(var_x < 5 && var_y < 5)
        return false;
    // remove random trajectory
    if( var_x > 100 || var_y > 100 )
        return false;

    for(int i = 2; i < size; i++) {
        float temp_x = track[i].x - track[i-1].x;
        float temp_y = track[i].y - track[i-1].y;
        length += sqrt(temp_x*temp_x+temp_y*temp_y);
    }
    // remove too short trajectory
    if(length<40) { //70
        return false;
    }

    // check the uniformality of the trajectory
    float len_thre = length*0.4;
    int nValidSeg = 0;
    std::vector<float> hod(4, 0);  // histogram of segment directions
    for( int i = 2; i < size; i++ ) {
        float temp_x = track[i].x - track[i-1].x;
        float temp_y = track[i].y - track[i-1].y;
        float temp_dist = sqrt(temp_x*temp_x + temp_y*temp_y);
        if( temp_dist > len_thre )
            return false;

        if (temp_dist < 0.1) {
            //hod[8]++;
        } else {
            int degree = (int)cv::fastAtan2(temp_y, temp_x);
            hod[degree/90]++;
            nValidSeg++;
        }
    }

    // check the straightness of the trajectory
    std::transform(hod.begin(), hod.end(), hod.begin(), std::bind2nd(std::divides<float>(), nValidSeg));
    float entropy = 0.0f;
    for (std::vector<float>::iterator it=hod.begin(); it!=hod.end(); it++) {
        if (std::fabs(*it) < 1e-4) {
            entropy += 0;
        } else {
            entropy += (*it) * std::log10(*it);
        }
    }
    entropy *= -1.0f;
    /*if (entropy > 0.45) {
        return false;
    }*/

    float directLen = cv::norm(track[1]-track[size-1]);
    if ((directLen/length<0.30f) || (directLen<10.0f)) {
        return false;
    }

    // trajectory of the door
    cv::Point2f firstPoint = track[1];
    cv::Point2f thirdPoint = track[3];
    //cv::Point2f lastPoint = track[size-1];
    float angle1 = cv::fastAtan2(firstPoint.y-thirdPoint.y, thirdPoint.x-firstPoint.x);
    //float angle2 = cv::fastAtan2(thirdPoint.y-lastPoint.y, lastPoint.x-thirdPoint.x);
    if (std::fabs(angle1-baseline_orient)<6.0) {
        return false;
    }

    return true;
}

void Counter::showResultImage(cv::Mat &img, int onPassengers, int offPassengers, std::string gps, std::string speed)
{
    int vOffset = 40;
    cv::Mat textImage = img.clone();
    std::ostringstream oss;

    oss << "BOARD: " << onPassengers;
    cv::putText(textImage, oss.str(), cv::Point(420,vOffset+0), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 255, 0));
    oss.str("");
    oss << "ALIGHT: " << offPassengers;
    cv::putText(textImage, oss.str(), cv::Point(420,vOffset+20), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 255, 0));

    oss.str("");
    oss << gps.substr(0,2) << " " << gps.substr(2,2) << "." << gps.substr(4,4);
    cv::putText(textImage, oss.str(), cv::Point(420,vOffset+60), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 0, 255));
    oss.str("");
    oss << gps.substr(8,2) << " " << gps.substr(10,2) << "." << gps.substr(12,4);
    cv::putText(textImage, oss.str(), cv::Point(420,vOffset+80), cv::FONT_HERSHEY_DUPLEX, 0.7, cv::Scalar(0, 0, 255));

    cv::addWeighted(img, 0.01, textImage, 0.99, 0.0, img);

    cv::imshow(windowName, img);
    //cv::cvtColor(image, image, CV_BGR2RGB);
    //QImage img((const unsigned char*)(image.data), image.cols, image.rows, QImage::Format_RGB888);
    //emit sendImage(img);
}

int Counter::onOrOffStartEnd(const std::vector<cv::Point2f>& track)
{
    std::vector<cv::Point2f>::const_iterator iSeg = track.begin();
    iSeg++;  // skip the time element
    cv::Point2f point0 = *iSeg;
    cv::Point2f point1 = track[track.size()-1];
    float orient = cv::fastAtan2(-(point1.y-point0.y), point1.x-point0.x); // -because opencv uses different xoy coordinate
    if (orient>30+baseline_orient && orient<170+baseline_orient) {//30 170
        return 0;
    } else if (orient>200+baseline_orient && orient<315+baseline_orient) {
        return 1;
    } else if (orient<baseline_orient-5){
        return 1;
    }else {
        return 2;
    }
}

int Counter::onOrOffHistOrient(const std::vector<cv::Point2f>& track)
{
    std::vector<float> orient_hist(3,0);
    std::vector<cv::Point2f>::const_iterator iSeg = track.begin();
    iSeg++;  // skip the time element
    cv::Point2f point0 = *iSeg;
    for (++iSeg; iSeg!= track.end(); iSeg++)
    {
        cv::Point2f point1 = *iSeg;
        float orient = cv::fastAtan2(-(point1.y-point0.y), point1.x-point0.x); // -because opencv uses different xoy coordinate
        float dist = cv::norm(point1-point0);
        if (orient>30+baseline_orient && orient<170+baseline_orient) {
            orient_hist[0] += dist;
        } else if (orient>200+baseline_orient && orient<315+baseline_orient) {
            orient_hist[1] += dist;
        } else if (orient < baseline_orient-5) {
            orient_hist[1] += dist;
        } else{
            orient_hist[2] += dist;
        }
        point0 = point1;
    }

    std::vector<float>::iterator iMax = std::max_element(orient_hist.begin(), orient_hist.end());
    return (int)std::distance(orient_hist.begin(), iMax);
}

// Get on (1)/off(0), 2 otherwise
int Counter::onOroff(const std::vector<cv::Point2f>& track)
{
    //return onOrOffHistOrient(track);
    return onOrOffStartEnd(track);
}

// Get the majority on/off according to the the tracking points
int Counter::getMajorityDirection(const std::list<std::vector<cv::Point2f> >& trajectories, int time)
{
    std::vector<int> orient_hist(3,0);
    float mean_x=0.0f, mean_y=0.0f;
    float var_x=0.0f, var_y=0.0f;
    float length=0.0f;
    std::list<std::vector<cv::Point2f> >::const_iterator iTrack = trajectories.begin();
    for (; iTrack!=trajectories.end(); ++iTrack) {
        if ((iTrack->at(0).x==time) && isValidTrack(*iTrack, mean_x, mean_y, var_x, var_y, length)) {
            int onoff = onOroff(*iTrack);
            orient_hist[onoff]++;
        }
    }

    std::vector<int>::iterator iMax = std::max_element(orient_hist.begin(), orient_hist.end());
    if (*iMax==0) return 2;
    else return (int)std::distance(orient_hist.begin(), iMax);
}

// onOrOff: boarding:1 or alighting:0
int Counter::updateModel(std::vector<std::list<int> >& models, const std::vector<cv::Point2f> &track, int onOrOff)
{

    float d_min = 1.1f;
    std::vector<std::list<int> >::iterator it_min;

    int start = track[0].x;   // start time of the track
    int min_track_group = (onOrOff==0 ? min_track_group_off : min_track_group_on);
    float t_threshold = (onOrOff==0 ? t_threshold_off : t_threshold_on);

    std::vector<std::list<int> >::iterator iModel = models.begin();
    for (; iModel!=models.end(); iModel++) {
        float ds = 1.0f;   // min distance to this model
        std::vector<float> vecds;
        std::list<int>::iterator iTrack = iModel->begin();
        for (; iTrack!=iModel->end(); iTrack++) {
            int iStart = *iTrack;
            float ds_track = 1.0f;   // min distance to a track in this group
            if (fabs(start-iStart) < t_min) {
                ds_track = 0.0;
            }
            else if (fabs(start-iStart) > t_max) {
                ds_track = 1.0;
            }
            else {
                ds_track = (fabs(start-iStart)-t_min)/(t_max-t_min);
            }

            vecds.push_back(ds_track);
        }

        std::sort(vecds.begin(), vecds.end());
        ds = vecds.at(vecds.size()/2);

        if (ds <= d_min) {
            d_min = ds;
            it_min = iModel;
        }

    }

    if (d_min < t_threshold) {
        it_min->push_back(start);
    }
    else {
        std::list<int> ls(1, start);
        models.push_back(ls);
    }

    //return models.size();
    int passengers = 0;
    iModel = models.begin();
    for (; iModel!=models.end(); iModel++)
    {
        if ((int)(iModel->size())>=min_track_group)
        {
            passengers++;
        }
    }

    return passengers;

}
