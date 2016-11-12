#include "VidStream.hpp"

#define GETK(key, type) (def[key] ? def[key] : defaults[key]).as<type>()

void VidStream::open(const YAML::Node &def)
{
    YAML::Node defaults = Settings::get()["video"]["defaults"];

    // Init variables
    this->valid = false;
    this->first = true;
    this->id = GETK("id", int);
    this->name = GETK("name", std::string);
    this->rez = GETK("res", rez_t);
    this->sf = GETK("scale", double);


    if (GETK("on", int) == 0)
    {
        std::cout << "Camera {{ " << this->name << " }} is off." << std::endl;
        return;
    }

    // Init VideoCapture
    std::cout << "Opening Camera {{ " << this->name << " }} ... ";
    this->cap.open(this->id);
    if (!this->cap.isOpened())
    {
        std::cout << "FAIL" << std::endl;
        return;
    }
    this->cap.set(cv::CAP_PROP_FRAME_WIDTH, this->rez[0]);
    this->cap.set(cv::CAP_PROP_FRAME_HEIGHT, this->rez[1]);
    this->valid = true;
    std::cout << "OK" << std::endl;
}

#undef GETK

void VidStream::t_loop()
{
    using namespace std::chrono_literals;
    cv::Mat frame, avg;
    double contrib = Settings::get()["video"]["contrib"].as<double>();
    // Capture loop
    while(this->running)
    {
        // Capture frame
        this->cap >> frame;
        if (this->first) avg = frame.clone();
        // Running exponential average
        cv::addWeighted(frame, contrib, avg, 1 - contrib, 0, avg);
        // Update stored, timestampped frame
        std::lock_guard<std::mutex> lock(this->frame_mutex);
        avg.copyTo(this->frame.frame);
        this->frame.timestamp = clk_t::now();
        if (this->first) this->first = false;
        std::this_thread::sleep_for(10ms);
    }
}

bool VidStream::getFrame(ts_frame &frame)
{
    if (!this->running || this->first)
    {
        return false;
    }
    std::lock_guard<std::mutex> lock(this->frame_mutex);
    frame.timestamp = this->frame.timestamp;
    resize(this->frame.frame, frame.frame, cv::Size(), this->sf, this->sf, cv::INTER_AREA);
    return true;
}

void VidStream::start()
{
    if (!this->valid)
    {
        std::cout << "The source {{ " << this->name << " }} is invalid." << std::endl;
        return;
    }
    if (this->running)
    {
        std::cout << "An instance of {{ " << this->name << " }} is already running." << std::endl;
        return;
    }
    this->running = true;
    this->t_vidstream = std::thread([=] { t_loop(); });
}
