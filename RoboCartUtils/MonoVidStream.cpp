#include "VidStream.hpp"

#ifdef AVG_ON
#define FRAME_ avg
#else
#define FRAME_ tmp
#endif

template <>
void VidStream<1>::capture_loop()
{
	ts_frame<1> tmp;
#ifdef AVG_ON
	double avg_weight = 1 - frame_weight;
	ts_frame<1> avg;
#endif
	// Capture loop
	while(running)
	{
		timestamp_t start = clk_t::now();
		// Capture frame --> transition to doing this in parallel (asynch)
		bool success = cap[0].grab();
		if (!success)
		{
			// Failed ... ???
			continue;
		}
		tmp.timestamp = clk_t::now();

		success = cap[0].retrieve(tmp.frame[0]);
#ifdef AVG_ON
		// Init properly if first frame
		if (first) {
			tmp.frame[0].copyTo(avg.frame[0]);
		}
		// Running exponential average
		cv::addWeighted(tmp.frame[0], frame_weight, avg.frame[0], avg_weight, 0, avg.frame[0]);
#endif
		std::lock_guard<std::mutex> lock(frame_mutex);

		FRAME_.frame[0].copyTo(frame.frame[0]);

		timestamp_t end = clk_t::now();
		milliseconds diff = duration_cast<milliseconds>(end - start);
		avg_time_ = avg_time_ * AVG_OLD + diff.count() * AVG_NEW;
		std::this_thread::sleep_for(10ms);
		if (first) first = false;
	}
}

template<>
bool VidStream<1>::getFrame(ts_frame<1> &frame_out)
{
	if (!running || first)
	{
		return false;
	}
	std::lock_guard<std::mutex> lock(frame_mutex);
	frame_out.timestamp = frame.timestamp;
	cv::Mat tmp;
#ifdef BLUR_ON
	cv::GaussianBlur(frame.frame[0], frame.frame[0],
			cv::Size( blur_radius, blur_radius ), 0, 0 );
#endif
	resize(frame.frame[0], frame_out.frame[0],
		cv::Size(), scaling_factor, scaling_factor, cv::INTER_AREA);
	return true;
}
