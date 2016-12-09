/** Core **/
#include <iostream>
#include <string>
#include <chrono>
#include <map>
#include <stdlib.h>

/** OpenCV **/
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

/** Other **/
#include "VidStream.hpp"
#include "StereoVision.hpp"
#include "OccupancyGrid.hpp"

using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

int main (int argc, char** argv )
{
    // Open up cameras
    std::cout << std::endl << "Opening VideoCapture streams" << std::endl;
    VidStream<2> cam;

    if (!cam.is_valid())
    {
    	std::cout << std::endl << "Invalid sources, exiting" << std::endl;
    	return 0;
    }

    std::cout << std::endl << "Entering main loop" << std::endl;

    ts_frame<2> frames;

    cam.start();
    std::string tm_disp;
    std::string tm_cap;
    cv::Mat disp_vis, disp_og, image3d;
    char name_l[16];
    char name_r[17];
    const char* fmt_l = "data/left%.2i.jpg";
    const char* fmt_r = "data/right%.2i.jpg";
    int i = 1;
    for (;;)
    {
        bool ready = cam.getFrame(frames);
        if (ready)
        {
        	imshow("Left", frames.frame[0]);
            imshow("Right", frames.frame[1]);
        }
        char k = (char) cv::waitKey(10);
        if (k == 'q') break;
        else if (k == 's' && ready)
        {
        	sprintf(name_l, fmt_l, i);
        	sprintf(name_r, fmt_r, i);
        	imwrite(name_l, frames.frame[0]);
        	imwrite(name_r, frames.frame[1]);
        	std::cout << "Saved image pair no. " << i << std::endl;
        	i++;
        }
    }
    cam.stop();

    return 0;
}
