#ifndef __VID_STREAM_H
#define __VID_STREAM_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <iostream>
#include <chrono>
#include <array>
#include <vector>
#include <cstdlib>

#include <thread>
#include <mutex>

#include <yaml-cpp/yaml.h>
#include "settings.hpp"

#define AVG_ON

typedef std::array<int, 2> rez_t;

using clk_t =  std::chrono::steady_clock;
using timestamp_t = std::chrono::time_point<clk_t>;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

template<std::size_t N>
struct ts_frame {
    std::array<cv::Mat, N> frame;
    timestamp_t timestamp;
};

template<std::size_t N>
using ts_frame = struct ts_frame<N>;

template<std::size_t N>
class VidStream {
private:
    // Settings
    bool valid;
    double scaling_factor;
    rez_t base_rez;
    // Video Source
    std::array<cv::VideoCapture, N> cap;
    // Frame Buffer
    ts_frame<N> frame;
    std::mutex frame_mutex;
    // Thread Loop
    bool running;
    bool first;
    void capture_loop()
    {
        using namespace std::chrono_literals;
        double frame_weight = Settings::get()["video"]["contrib"].as<double>();
        double avg_weight = 1 - frame_weight;
        ts_frame<N> tmp;
#ifdef AVG_ON
        ts_frame<N> avg;
#endif
        // Capture loop
        while(running)
        {
            timestamp_t start = clk_t::now();
            // Capture frame --> transition to doing this in parallel (asynch)
            bool success = true;
            int i;
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
                timestamp_t end = clk_t::now();
                milliseconds diff = duration_cast<milliseconds>(end - start);
    	        avg_time_ = avg_time_ * 0.99 + diff.count() * 0.01;
            }
            std::this_thread::sleep_for(10ms);
            if (first) first = false;
        }
    }
    double avg_time_;
    std::thread t_vidstream;
public:
    VidStream(int* src_ids)
    {
        YAML::Node video_settings = Settings::get()["video"];

        // Init variables
        valid = false;
        first = true;
        running = false;
        base_rez = video_settings["src-rez"].as<rez_t>();
        scaling_factor = video_settings["scale"].as<double>();

        // Init VideoCapture
        for(int i = 0; i < N; i++)
        {
            std::cout << "Opening Camera {{ " << video_settings["source"][i].as<std::string>() << " }} " << src_ids[i] << " ... ";
            cap[i].open(src_ids[i]);
            frame.frame[i].create(base_rez[1], base_rez[0], CV_8UC3);
            if (!cap[i].isOpened())
            {
                std::cout << "FAIL" << std::endl;
                return;
            }
            cap[i].set(cv::CAP_PROP_FRAME_WIDTH, base_rez[0]);
            cap[i].set(cv::CAP_PROP_FRAME_HEIGHT, base_rez[1]);
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
        int i;
        if (scaling_factor < 1.0)
            for(i = 0; i < N; i++)
                resize(frame.frame[i], frame_out.frame[i],
                    cv::Size(), scaling_factor, scaling_factor, cv::INTER_AREA);
        else
            for(i = 0; i < N; i++)
                frame.frame[i].copyTo(frame_out.frame[i]);
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
