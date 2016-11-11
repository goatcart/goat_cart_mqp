/** Core **/
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <thread>
#include <chrono>
#include <stdio.h>

/** OpenCV **/
#include <opencv2/core.hpp>
//#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/videoio.hpp>
//#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
//#include <opencv2/stereo.hpp>

/** Other **/
#include <yaml-cpp/yaml.h>
#include "defaults.h"

using namespace cv;

/*
%YAML:1.0
---
name: "StereoMatcher.SGBM"
blockSize: 5
minDisparity: 0
numDisparities: 128
disp12MaxDiff: 10
speckleRange: 7
speckleWindowSize: 60
P1: 600
P2: 2048
preFilterCap: 1
uniquenessRatio: 0
mode: 1

*/

#define P_base(imChannels, blockSize) (8 * imChannels * blockSize * blockSize)
#define FPS 1

struct AppParams {
float scaleFactor;
    int blockSize;
    int numDisparities;
    int disp12MaxDiff;
    int speckleRange;
    int speckleWindowSize;
    int preFilterCap;
    int uniquenessRatio;
};

void loadSettings(YAML::Node *pStar, struct AppParams *s)
{
    YAML::Node p = *pStar;
    YAML::Node node;

    node = p["scaleFactor"];
    s->scaleFactor = node.IsScalar() ? node.as<float>() : DEFAULT_SCALE_FACTOR;

    node = p["blockSize"];
    s->blockSize = node.IsScalar() ? node.as<int>() : DEFAULT_BLOCK_SIZE;

    node = p["numDisparities"];
    s->numDisparities = node.IsScalar() ? node.as<int>() : DEFAULT_DISP_COUNT;

    node = p["disp12MaxDiff"];
    s->disp12MaxDiff = node.IsScalar() ? node.as<int>() : DEFAULT_D12_MAX_DIFF;

    node = p["speckleRange"];
    s->speckleRange = node.IsScalar() ? node.as<int>() : DEFAULT_SPECKLE_RANGE;

    node = p["speckleWindowSize"];
    s->speckleWindowSize = node.IsScalar() ? node.as<int>() : DEFAULT_SPECKE_WS;

    node = p["preFilterCap"];
    s->preFilterCap = node.IsScalar() ? node.as<int>() : DEFAULT_PF_CAP;

    node = p["uniquenessRatio"];
    s->uniquenessRatio = node.IsScalar() ? node.as<int>() : DEFAULT_UNIQ_RATIO;
}

int initCam(VideoCapture *cam, int id, int fps, int w, int h)
{
    std::cout << "Opening Camera {{ " << id << " }} ... ";
    cam->open(id);
    if (!cam->isOpened())
    {
        std::cout << "FAIL" << std::endl;
        return -1;
    }
//    cam->set(CAP_PROP_FPS,          fps);
    cam->set(CAP_PROP_MODE,         CAP_MODE_GRAY);
    cam->set(CAP_PROP_FRAME_WIDTH,  w);
    cam->set(CAP_PROP_FRAME_HEIGHT, h);
    std::cout << "OK" << std::endl;
    return 0;
}

int main (int argc, char** argv )
{
    std::cout << "Loading params\n";
    struct AppParams pSettings;
    YAML::Node params = YAML::LoadFile("params.yml");
    YAML::Node cams_s = params["cams"];
    YAML::Node matcher_s = params["matcher"];

    loadSettings(&matcher_s, &pSettings);

    // Open up cameras
    VideoCapture cam_l, cam_r;
    int fps = cams_s["framerate"].as<int>();
    int w = cams_s["size"][0].as<int>();
    int h = cams_s["size"][1].as<int>();
    if (initCam(&cam_l, cams_s["id"][0].as<int>(), fps, w, h) == -1) return -1;
    if (initCam(&cam_r, cams_s["id"][1].as<int>(), fps, w, h) == -1) return -1;

    std::cout << "Creating StereoMatcher" << std::endl;
    Ptr<StereoSGBM> matcher = StereoSGBM::create(0, pSettings.numDisparities, pSettings.blockSize);
    matcher->setP1(P_base(1, pSettings.blockSize));
    matcher->setP2(4 * P_base(1, pSettings.blockSize));
    matcher->setDisp12MaxDiff(pSettings.disp12MaxDiff);
    matcher->setSpeckleRange(pSettings.speckleRange);
    matcher->setSpeckleWindowSize(pSettings.speckleWindowSize);
    matcher->setPreFilterCap(pSettings.preFilterCap);
    matcher->setUniquenessRatio(pSettings.uniquenessRatio);

    std::cout << "Entering main loop" << std::endl;

    Mat frameL, frameR, scaleR, scaleL;
    Mat disp, disp_vis;
    for (;;)
    {
        cam_l >> frameL;
        cam_r >> frameR;
        resize(frameL, scaleL, Size(), pSettings.scaleFactor, pSettings.scaleFactor);
        resize(frameR, scaleR, Size(), pSettings.scaleFactor, pSettings.scaleFactor);
        matcher->compute(scaleL, scaleR, disp);
        ximgproc::getDisparityVis(disp, disp_vis, 2);
        imshow("Disparity Map", disp_vis);
        if ((char) waitKey(1) == 'q') break;
    }

    return 0;
}
