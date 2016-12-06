/** Core **/
#include <iostream>
#include <string>
#include <chrono>
#include <map>

/** OpenCV **/
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

/** Other **/
#include "VidStream.hpp"

using namespace cv;
using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

#define P_base(imChannels, blockSize) (8 * imChannels * blockSize * blockSize)
#define CALC_DISP(width, scale) (((int) (width * scale / 8) + 15) & -16)

struct AppParams {
    double scaleFactor;
    int alg;
    int blockSize;
    int numDisparities;
    int disp12MaxDiff;
    int speckleRange;
    int speckleWindowSize;
    int preFilterCap;
    int uniquenessRatio;
};

void loadSettings(struct AppParams &s)
{
	FileStorage fs("params.yml", FileStorage::READ);
	FileNode m = fs["matcher"];

    s.scaleFactor = (float) m["scaleFactor"];
    s.alg = (int) m["alg"];

    if (s.alg == 1) s.blockSize = 3;
    else if (s.scaleFactor < 0.99) s.blockSize = 7;
    else s.blockSize = 15;

    int default_width = (int) fs["video"]["src-rez"][0];
    s.numDisparities = CALC_DISP(default_width, s.scaleFactor);
    s.disp12MaxDiff = (int) m["disp12MaxDiff"];
    s.speckleRange = (int) m["speckleRange"];
    s.speckleWindowSize = (int) m["speckleWindowSize"];
    s.preFilterCap = (int) m["preFilterCap"];
    s.uniquenessRatio = (int) m["uniquenessRatio"];
}

Ptr<StereoMatcher> initMatcher(struct AppParams &pSettings)
{
    Ptr<StereoBM> matcher = StereoBM::create(pSettings.numDisparities, pSettings.blockSize);
    //matcher->setDisp12MaxDiff(pSettings.disp12MaxDiff);
    //matcher->setSpeckleRange(pSettings.speckleRange);
    //matcher->setSpeckleWindowSize(pSettings.speckleWindowSize);
    matcher->setPreFilterCap(pSettings.preFilterCap);
    matcher->setUniquenessRatio(pSettings.uniquenessRatio);
    return matcher;
}



int main (int argc, char** argv )
{
    struct AppParams pSettings;

    loadSettings(pSettings);

    // Open up cameras
    std::cout << std::endl << "Opening VideoCapture streams" << std::endl;
    VidStream<2> cam;

    std::cout << std::endl << "Creating StereoMatcher" << std::endl;
    Ptr<StereoMatcher> matcher_l = initMatcher(pSettings);
    Ptr<ximgproc::DisparityWLSFilter> wls_filter = ximgproc::createDisparityWLSFilter(matcher_l);
    Ptr<StereoMatcher> matcher_r = ximgproc::createRightMatcher(matcher_l);
    wls_filter->setLambda(8000);
    wls_filter->setSigmaColor(1.0);

    std::cout << std::endl << "Entering main loop" << std::endl;

    ts_frame<2> frames;
    Mat disp_l, disp_r, disp_f, disp_vis;
    cam.start();
    std::string tm_cap;
    double avg_time = 0;
#define DV disp_vis
    for (;;)
    {
        bool ready = cam.getFrame(frames);
        if (ready)
        {
            time_point<Clock> start = Clock::now();
            matcher_l->compute(frames.frame[0], frames.frame[1], disp_l);
            matcher_r->compute(frames.frame[1], frames.frame[0], disp_r);
            wls_filter->filter(disp_l, frames.frame[0], disp_f, disp_r);
            ximgproc::getDisparityVis(disp_f, disp_vis, 1.75);
            time_point<Clock> end = Clock::now();
            // Calc + disp filter time
            milliseconds diff = duration_cast<milliseconds>(end - start);
	        avg_time = avg_time * 0.75 + diff.count() * 0.25;
            tm_cap = std::to_string((int) (avg_time + 0.5));
            putText(DV, tm_cap, Point(10, 100), FONT_HERSHEY_PLAIN, 2.0, 	Scalar(255.0, 255.0, 255.0), 2);
            imshow("Disparity Map", DV);
            imshow("Left", frames.frame[0]);
            imshow("Right", frames.frame[1]);
        }
        if ((char) waitKey(50) == 'q') break;
    }
    cam.stop();

    return 0;
}
