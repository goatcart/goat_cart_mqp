/** Core **/
#include <iostream>
#include <string>
#include <chrono>
#include <map>

/** OpenCV **/
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

/** Other **/
#include <yaml-cpp/yaml.h>
#include "defaults.h"
#include "settings.hpp"
#include "VidStream.hpp"

using namespace cv;

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

void loadSettings(struct AppParams *s)
{
    YAML::Node p = Settings::get();
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
}

#define GETK(key, type) (t1[key] ? t1[key] : t2[key]).as<type>()

int main (int argc, char** argv )
{
    using namespace std::chrono_literals;
    /**/
    std::cout << std::endl << "Loading params" << std::endl;
    YAML::Node cams_s = Settings::get()["video"]["sources"];
    struct AppParams pSettings;

    //loadSettings(&pSettings);

    // Open up cameras
    std::cout << std::endl << "Opening VideoCapture streams" << std::endl;
    std::map<std::string, VidStream> caps;
    for (int i = 0; i < cams_s.size(); i++)
    {
        caps[cams_s[i]["name"].as<std::string>()].open(cams_s[i], VidStream::filtGray | VidStream::filtResize);
    }

    std::cout << std::endl << "Entering main loop" << std::endl;

    ts_frame frame;
    VidStream *cam_l = &caps[Settings::get()["cam"].as<std::string>()];
    cam_l->start();
    for (;;)
    {
        bool ready = cam_l->getFrame(frame);
        if (ready) imshow("Cam", frame.frame);
        if ((char) waitKey(30) == 'q') break;
    }
    cam_l->stop();
    /**/
    return 0;
}
