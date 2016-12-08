/*
 * types.hpp
 *
 *  Created on: Dec 6, 2016
 *      Author: plong
 */

#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <opencv2/core.hpp>
#include <chrono>

using clk_t =  std::chrono::steady_clock;
using timestamp_t = std::chrono::time_point<clk_t>;

template<std::size_t N>
struct ts_frame_ {
    cv::Mat frame[N];
    timestamp_t timestamp;
};

template<std::size_t N>
using ts_frame = struct ts_frame_<N>;



#endif /* TYPES_HPP_ */
