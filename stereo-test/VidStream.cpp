#include "VidStream.hpp"

#define GETK(key, type) (def[key] ? def[key] : defaults[key]).as<type>()

void VidStream::open(const YAML::Node &def, int filters)
{
    YAML::Node defaults = Settings::get()["video"]["defaults"];

    // Init variables
    valid = false;
    first = true;
    id = GETK("id", int);
    name = GETK("name", std::string);
    rez = GETK("res", rez_t);
    sf = GETK("scale", double);


    if (GETK("on", int) == 0)
    {
        std::cout << "Camera {{ " << name << " }} is off." << std::endl;
        return;
    }

    // Init VideoCapture
    std::cout << "Opening Camera {{ " << name << " }} ... ";
    cap.open(id);
    if (!cap.isOpened())
    {
        std::cout << "FAIL" << std::endl;
        return;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, rez[0]);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, rez[1]);
    valid = true;
    std::cout << "OK" << std::endl;
}

#undef GETK

void VidStream::t_loop()
{
    using namespace std::chrono_literals;
    cv::Mat frame, avg;
    double contrib = Settings::get()["video"]["contrib"].as<double>();
    // Capture loop
    while(running)
    {
        // Capture frame
        cap >> frame;
        if (first) avg = frame.clone();
        // Running exponential average
        cv::addWeighted(frame, contrib, avg, 1 - contrib, 0, avg);
        // Update stored, timestampped frame
        std::lock_guard<std::mutex> lock(frame_mutex);
        avg.copyTo(_frame.frame);
        _frame.timestamp = clk_t::now();
        std::this_thread::sleep_for(10ms);
        if (first) first = false;
    }
}

bool VidStream::getFrame(ts_frame &frame)
{
    if (!running || first)
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(frame_mutex);
    frame.timestamp = _frame.timestamp;
    if (filters & filtResize) resize(_frame.frame, frame.frame, cv::Size(), sf, sf, cv::INTER_AREA);
    else _frame.frame.copyTo(frame.frame);
    if (filters & filtGray) cvtColor(frame.frame, frame.frame, cv::COLOR_BGR2GRAY);
    return true;
}

void VidStream::start()
{
    if (!valid)
    {
        std::cout << "The source {{ " << name << " }} is invalid." << std::endl;
        return;
    }
    if (running)
    {
        std::cout << "An instance of {{ " << name << " }} is already running." << std::endl;
        return;
    }
    running = true;
    t_vidstream = std::thread([=] { t_loop(); });
}
