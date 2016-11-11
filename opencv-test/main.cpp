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
#define CALC_DISP(width, scale) (((int) (width * scale / 8) + 15) & -16)

struct AppParams {
float scaleFactor;
    int alg;
    int blockSize;
    int numDisparities;
    int disp12MaxDiff;
    int speckleRange;
    int speckleWindowSize;
    int preFilterCap;
    int uniquenessRatio;
    int useHH;
};

void loadSettings(YAML::Node *pStar, struct AppParams *s)
{
    YAML::Node p = *pStar;
    YAML::Node m = p["matcher"];
    YAML::Node node;

    node = m["scaleFactor"];
    s->scaleFactor = node.IsScalar() ? node.as<float>() : DEFAULT_SCALE_FACTOR;

    node = m["alg"];
    s->alg = node.IsScalar() ? node.as<int>() : DEFAULT_ALG;

    if (s->alg == 1) s->blockSize = 3;
    else if (s->scaleFactor < 0.99) s->blockSize = 7;
    else s->blockSize = 15;

    s->numDisparities = CALC_DISP(p["cams"]["size"][0].as<int>(), s->scaleFactor);

    node = m["disp12MaxDiff"];
    s->disp12MaxDiff = node.IsScalar() ? node.as<int>() : DEFAULT_D12_MAX_DIFF;

    node = m["speckleRange"];
    s->speckleRange = node.IsScalar() ? node.as<int>() : DEFAULT_SPECKLE_RANGE;

    node = m["speckleWindowSize"];
    s->speckleWindowSize = node.IsScalar() ? node.as<int>() : DEFAULT_SPECKE_WS;

    node = m["preFilterCap"];
    s->preFilterCap = node.IsScalar() ? node.as<int>() : DEFAULT_PF_CAP;

    node = m["uniquenessRatio"];
    s->uniquenessRatio = node.IsScalar() ? node.as<int>() : DEFAULT_UNIQ_RATIO;

    node = m["useHH"];
    s->useHH = node.IsScalar() ? node.as<int>() : 0;
}

int initCam(VideoCapture *cam, int id, int w, int h)
{
    std::cout << "Opening Camera {{ " << id << " }} ... ";
    cam->open(id);
    if (!cam->isOpened())
    {
        std::cout << "FAIL" << std::endl;
        return -1;
    }
    cam->set(CAP_PROP_MODE,         CAP_MODE_GRAY);
    cam->set(CAP_PROP_FRAME_WIDTH,  w);
    cam->set(CAP_PROP_FRAME_HEIGHT, h);
    std::cout << "OK" << std::endl;
    return 0;
}

Ptr<StereoMatcher> initMatcher(struct AppParams *pSettings)
{
    Ptr<StereoBM> matcher = StereoBM::create(pSettings->numDisparities, pSettings->blockSize);
    matcher->setDisp12MaxDiff(pSettings->disp12MaxDiff);
    matcher->setSpeckleRange(pSettings->speckleRange);
    matcher->setSpeckleWindowSize(pSettings->speckleWindowSize);
    matcher->setPreFilterCap(pSettings->preFilterCap);
    matcher->setUniquenessRatio(pSettings->uniquenessRatio);
    return matcher;
}

Ptr<StereoMatcher> initMatcherSG(struct AppParams *pSettings)
{
    Ptr<StereoSGBM> matcher = StereoSGBM::create(0, pSettings->numDisparities, pSettings->blockSize);
    matcher->setP1(P_base(1, pSettings->blockSize));
    matcher->setP2(4 * P_base(1, pSettings->blockSize));
    //matcher->setDisp12MaxDiff(pSettings->disp12MaxDiff);
    //matcher->setSpeckleRange(pSettings->speckleRange);
    //matcher->setSpeckleWindowSize(pSettings->speckleWindowSize);
    matcher->setPreFilterCap(pSettings->preFilterCap);
    matcher->setMode(pSettings->useHH);
    return matcher;
}

int main (int argc, char** argv )
{
    std::cout << std::endl << "Loading params" << std::endl;
    struct AppParams pSettings;
    YAML::Node params = YAML::LoadFile("params.yml");
    YAML::Node cams_s = params["cams"];

    loadSettings(&params, &pSettings);

    // Open up cameras
    std::cout << std::endl << "Opening VideoCapture streams" << std::endl;
    VideoCapture cam_l, cam_r;
    int w = cams_s["size"][0].as<int>();
    int h = cams_s["size"][1].as<int>();
    if (initCam(&cam_l, cams_s["id"][0].as<int>(), w, h) == -1) return -1;
    if (initCam(&cam_r, cams_s["id"][1].as<int>(), w, h) == -1) return -1;

    std::cout << std::endl << "Creating StereoMatcher" << std::endl;
    Ptr<StereoMatcher> matcher_l;
    if (pSettings.alg) matcher_l = initMatcherSG(&pSettings);
    else matcher_l = initMatcher(&pSettings);
    Ptr<ximgproc::DisparityWLSFilter> wls_filter = ximgproc::createDisparityWLSFilter(matcher_l);
    Ptr<StereoMatcher> matcher_r = ximgproc::createRightMatcher(matcher_l);
    wls_filter->setLambda(8000);
    wls_filter->setSigmaColor(1.0);

    std::cout << "[INFO] Nd = " << pSettings.numDisparities << ", Pb = " << P_base(1, pSettings.blockSize) << std::endl;

    std::cout << std::endl << "Entering main loop" << std::endl;

    Mat frameL, frameR, scaleR, scaleL;
    Mat disp_l, disp_r, disp_f, disp_vis;
    for (;;)
    {
        cam_l >> frameL;
        cam_r >> frameR;
        resize(frameL, scaleL, Size(), pSettings.scaleFactor, pSettings.scaleFactor, INTER_AREA);
        resize(frameR, scaleR, Size(), pSettings.scaleFactor, pSettings.scaleFactor, INTER_AREA);
        matcher_l->compute(scaleL, scaleR, disp_l);
        matcher_r->compute(scaleR, scaleL, disp_r);
        wls_filter->filter(disp_l, scaleL, disp_f, disp_r);
        ximgproc::getDisparityVis(disp_f, disp_vis, 1.75);
        imshow("Disparity Map", disp_vis);
        if ((char) waitKey(1) == 'q') break;
    }

    return 0;
}
