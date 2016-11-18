#ifndef __VID_STREAM_H
#define __VID_STREAM_H

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <iostream>
#include <chrono>
#include <array>

#include <thread>
#include <mutex>

#include <yaml-cpp/yaml.h>
#include "settings.hpp"

typedef std::array<int, 2> rez_t;
using clk_t =  std::chrono::steady_clock;
using timestamp_t = std::chrono::time_point<clk_t>;

typedef struct {
    cv::Mat frame;
    timestamp_t timestamp;
} ts_frame;


class VidStream {
private:
    // Settings
    bool valid;
    int id;
    std::string name;
    double sf;
    rez_t rez;
    int filters;
    // Video Source
    cv::VideoCapture cap;
    // Frame
    ts_frame _frame;
    std::mutex frame_mutex;
    // Thread Loop
    bool running;
    bool first;
    void t_loop();
    std::thread t_vidstream;
public:
    enum vid_filters
    {
        filtResize = 1,
        filtGray = 2
    };
    VidStream(): valid(false), running(false) {}
    void open(const YAML::Node &, int filters = filtResize);
    bool getFrame(ts_frame &);
    void stop() { running = false; this->t_vidstream.join(); }
    void start();
};

#endif
