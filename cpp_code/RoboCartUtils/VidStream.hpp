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
protected:
	// Settings
	bool valid;
	double frame_weight;
	int blur_radius;
	double scaling_factor;
	bool gray;
	cv::Size base_rez;
	// Video Source
	cv::VideoCapture cap[N];
	// Frame Buffer
	ts_frame<N> frame;
	std::mutex frame_mutex;
	// Thread Loop
	bool running;
	bool first;
	void capture_loop();
	double avg_time_;
	std::thread t_vidstream;
public:
	VidStream() {
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
		gray = (1 == (int) video_settings["gray"]);

		// Init VideoCapture
		for (unsigned int i = 0; i < N; i++) {
			std::cout << "Opening Camera {{ "
					<< (std::string) video_settings["name"][i] << " }} ... ";
			cap[i].open((int) video_settings["src"][i]);
			frame.frame[i].create(base_rez, CV_8UC3);
			if (!cap[i].isOpened()) {
				std::cout << "FAIL" << std::endl;
				return;
			}
			cap[i].set(cv::CAP_PROP_FRAME_WIDTH, base_rez.width);
			cap[i].set(cv::CAP_PROP_FRAME_HEIGHT, base_rez.height);
			std::cout << "OK" << std::endl;
		}
		valid = true;
	}
	bool getFrame(ts_frame<N> &frame_out);
	void stop() {
		running = false;
		this->t_vidstream.join();
	}
	void start() {
		if (!valid) {
			std::cout << "One of the sources is invalid." << std::endl;
			return;
		}
		if (running) {
			std::cout << "An instance is already running." << std::endl;
			return;
		}
		running = true;
		t_vidstream = std::thread([=] {capture_loop();});
	}
	const double avg_time() {
		return avg_time_;
	}
	const bool is_valid() {
		return valid;
	}
};

#endif
