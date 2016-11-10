/** Core **/
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>

/** OpenCV **/
#include <opencv2/core.hpp>
//#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
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
    int framerate;
    int blockSize;
    int numDisparities;
    int minDisparity;
    int blurRadius;
    float scaleFactor;
    int numCams;
    int camL;
    int camR;
};

void DoTestCalc (VideoCapture *caps, struct AppParams *pSettings)
{
    Mat frame, blur, scale;
    caps[0] >> frame;
    GaussianBlur(frame, blur, Size(pSettings->blurRadius, pSettings->blurRadius), 0, 0);
    resize(blur, scale, Size(), pSettings->scaleFactor, pSettings->scaleFactor, INTER_AREA);
    std::stringstream title;
    title << "Frame (FPS: " << caps[0].get(CAP_PROP_FPS)  << ")";
    imshow(title.str(), scale);
}

int main (int argc, char** argv )
{
    std::cout << "Loading params\n";
    struct AppParams pSettings;
    YAML::Node params = YAML::LoadFile("params.yml");
    YAML::Node cams = params["cams"];
    YAML::Node node;

    node = params["framerate"];
    pSettings.framerate = node.IsScalar() ? node.as<int>() : DEFAULT_FRAMERATE;

    node = params["blockSize"];
    pSettings.blockSize = node.IsScalar() ? node.as<int>() : DEFAULT_BLOCK_SIZE;

    node = params["numDisparities"];
    pSettings.numDisparities = node.IsScalar() ? node.as<int>() : DEFAULT_DISPARITY_COUNT;

    node = params["minDisparity"];
    pSettings.minDisparity = node.IsScalar() ? node.as<int>() : DEFAULT_MIN_DISPARITIES;

    node = params["blurRadius"];
    pSettings.blurRadius = node.IsScalar() ? node.as<int>() : DEFAULT_BLUR_RADIUS;

    node = params["scaleFactor"];
    pSettings.scaleFactor = node.IsScalar() ? node.as<float>() : DEFAULT_SCALE_FACTOR;

    node = params["camL"];
    pSettings.camL = node.IsScalar() ? node.as<int>() : DEFAULT_LEFT_CAM_INDEX;

    node = params["camR"];
    pSettings.camR = node.IsScalar() ? node.as<int>() : DEFAULT_RIGHT_CAM_INDEX;

    // Open up cameras
    pSettings.numCams = cams.size();
    VideoCapture caps[pSettings.numCams];
    std::cout << "Opening Camera Streams (num = " << pSettings.numCams << ")" << std::endl;
    for (size_t i = 0; i < pSettings.numCams; i++)
    {
        std::cout << "Opening Camera {{ " << cams[i]["name"].as<std::string>() << " }} ... ";
        caps[i].open(cams[i]["id"].as<int>());
        if (!caps[i].isOpened())
        {
            std::cout << "FAIL" << std::endl;
            return -1;
        }
        caps[i].set(CAP_PROP_FPS,          pSettings.framerate);
        caps[i].set(CAP_PROP_MODE,         CAP_MODE_GRAY);
        caps[i].set(CAP_PROP_FRAME_WIDTH,  cams[i]["w"].as<int>());
        caps[i].set(CAP_PROP_FRAME_HEIGHT, cams[i]["h"].as<int>());
        std::cout << "OK" << std::endl;
    }

    std::cout << "Creating StereoMatcher" << std::endl;
    Ptr<StereoSGBM> matcher = StereoSGBM::create(pSettings.minDisparity, pSettings.numDisparities, pSettings.blockSize);
    matcher->setP1(P_base(1, pSettings.blockSize));
    matcher->setP2(4 * P_base(1, pSettings.blockSize));
    matcher->setDisp12MaxDiff(10);
    matcher->setSpeckleRange(7);
    matcher->setSpeckleWindowSize(60);
    matcher->setPreFilterCap(1);
    matcher->setUniquenessRatio(0);
    matcher->setMode(StereoSGBM::MODE_HH);

    std::cout << "Entering main loop" << std::endl;

    Mat frameL, frameR, blurL, blurR, scaleR, scaleL;
    Mat disp, disp8, disp_color;
    for (;;)
    {
        caps[pSettings.camL] >> frameL;
        caps[pSettings.camR] >> frameR;
        GaussianBlur(frameL, blurL, Size(pSettings.blurRadius, pSettings.blurRadius), 0, 0);
        resize(blurL, scaleL, Size(), pSettings.scaleFactor, pSettings.scaleFactor);
        GaussianBlur(frameR, blurR, Size(pSettings.blurRadius, pSettings.blurRadius), 0, 0);
        resize(blurR, scaleR, Size(), pSettings.scaleFactor, pSettings.scaleFactor);
        matcher->compute(scaleL, scaleR, disp);
        disp.convertTo(disp8, CV_8U, 255/(pSettings.numDisparities*16.));
        cvtColor(disp8, disp_color, CV_GRAY2RGB);
        imshow("Disparity Map", disp_color);
        if ((char) waitKey(1) == 'q') break;
    }

    return 0;
}
