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

    std::cout << std::endl << "Creating StereoMatcher" << std::endl;
    StereoVision depthMapper;

    std::cout << std::endl << "Creating OccupancyGrid" << std::endl;
    OccupancyGrid ogCalculator;

    std::cout << std::endl << "Entering main loop" << std::endl;

    ts_frame<2> frames;

    cam.start();
    std::string tm_disp;
    std::string tm_cap;
    cv::Mat disp_vis, disp_og_in, disp_og, image3d;
    for (;;)
    {
        bool ready = cam.getFrame(frames);
        if (ready)
        {
        	depthMapper.compute(frames, disp_vis);
        	normalize(disp_vis, disp_og_in, 0, 255, CV_MINMAX, CV_8UC1);
        	ogCalculator.compute(disp_og_in, disp_og, image3d);
        	tm_disp = std::to_string((int) (depthMapper.avg_time() + 0.5));
            tm_cap = std::to_string((int) (cam.avg_time() + 0.5));
            cv::putText(disp_vis, tm_disp, cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255.0, 255.0, 255.0), 2);
            cv::putText(disp_vis, tm_cap, cv::Point(10, 200), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255.0, 255.0, 255.0), 2);
            cv::imshow("Disparity Map", disp_og_in);
            cv::imshow("OG", disp_og);
            imshow("Left", frames.frame[0]);
            imshow("Right", frames.frame[1]);
        }
        if ((char) cv::waitKey(10) == 'q') break;
    }
    cam.stop();

    return 0;
}
