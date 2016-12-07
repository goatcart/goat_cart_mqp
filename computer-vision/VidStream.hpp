#ifndef __VID_STREAM_H
#define __VID_STREAM_H

#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <cstdlib>

#include <thread>
#include <mutex>

#include "settings.hpp"
#include "types.hpp"

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using namespace std::chrono_literals;

template<std::size_t N>
class VidStream {
private:
    // Settings
    bool valid;
    double frame_weight;
    int blur_radius;
    double scaling_factor;
    cv::Size base_rez;
    // Video Source
    cv::VideoCapture cap[N];
    // Frame Buffer
    ts_frame<N> frame;
    std::mutex frame_mutex;
    // Thread Loop
    bool running;
    bool first;
    void capture_loop()
    {
        ts_frame<N> tmp;
#ifdef AVG_ON
        double avg_weight = 1 - frame_weight;
        ts_frame<N> avg;
#endif
        // Capture loop
        while(running)
        {
            timestamp_t start = clk_t::now();
            // Capture frame --> transition to doing this in parallel (asynch)
            bool success = true;
            unsigned int i;
            for (i = 0; i < N; i++)
                success &= cap[i].grab();
            if (!success)
            {
                // Failed ... ???
                continue;
            }
            tmp.timestamp = clk_t::now();
            success = true;
            for (i = 0; i < N; i++)
            {
                success &= cap[i].retrieve(tmp.frame[i]);
#ifdef AVG_ON
                // Init properly if first frame
                if (first)
                    tmp.frame[i].copyTo(avg.frame[i]);
                // Running exponential average
                cv::addWeighted(tmp.frame[i], frame_weight, avg.frame[i], avg_weight, 0, avg.frame[i]);
#endif
            }
            std::lock_guard<std::mutex> lock(frame_mutex);
            for (i = 0; i < N; i++)
            {
#ifdef AVG_ON
                cvtColor(avg.frame[i], frame.frame[i], cv::COLOR_BGR2GRAY);
#else
                cvtColor(tmp.frame[i], frame.frame[i], cv::COLOR_BGR2GRAY);
#endif
            }

            timestamp_t end = clk_t::now();
            milliseconds diff = duration_cast<milliseconds>(end - start);
            avg_time_ = avg_time_ * AVG_OLD + diff.count() * AVG_NEW;
            std::this_thread::sleep_for(10ms);
            if (first) first = false;
        }
    }
    double avg_time_;
    std::thread t_vidstream;
public:
    VidStream()
    {
    	cv::FileStorage fs("params.yml", cv::FileStorage::READ);
    	cv::FileNode video_settings = fs["video"];

        // Init variables
        valid = false;
        first = true;
        running = false;
        video_settings["src-rez"] >> base_rez;
        scaling_factor = (double) video_settings["scale"];
        blur_radius = (int) video_settings["blur"];
        avg_time_ = 0;
        frame_weight = (double) video_settings["contrib"];

        // Init VideoCapture
        for(unsigned int i = 0; i < N; i++)
        {
            std::cout << "Opening Camera {{ " << (std::string) video_settings["name"][i] << " }} ... ";
            cap[i].open((int) video_settings["src"][i]);
            frame.frame[i].create(base_rez, CV_8UC3);
            if (!cap[i].isOpened())
            {
                std::cout << "FAIL" << std::endl;
                return;
            }
            cap[i].set(cv::CAP_PROP_FRAME_WIDTH, base_rez.width);
            cap[i].set(cv::CAP_PROP_FRAME_HEIGHT, base_rez.height);
            std::cout << "OK" << std::endl;
        }
        valid = true;
    }
    bool getFrame(ts_frame<N> &frame_out)
    {
        if (!running || first)
        {
            return false;
        }
        std::lock_guard<std::mutex> lock(frame_mutex);
        frame_out.timestamp = frame.timestamp;
        unsigned int i;
        cv::Mat tmp;
        for(i = 0; i < N; i++)
        {
#ifdef BLUR_ON
            cv::GaussianBlur( frame.frame[i], frame.frame[i],
                    cv::Size( blur_radius, blur_radius ), 0, 0 );
#endif
			resize( frame.frame[i], frame_out.frame[i],
				cv::Size(), scaling_factor, scaling_factor, cv::INTER_AREA);
        }
        return true;
    }
    void stop() { running = false; this->t_vidstream.join(); }
    void start()
    {
        if (!valid)
        {
            std::cout << "One of the sources is invalid." << std::endl;
            return;
        }
        if (running)
        {
            std::cout << "An instance is already running." << std::endl;
            return;
        }
        running = true;
        t_vidstream = std::thread([=] { capture_loop(); });
    }
    const double avg_time() { return avg_time_; }
};

#endif
