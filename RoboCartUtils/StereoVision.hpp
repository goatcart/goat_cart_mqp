/*
 * StereoVision.h
 *
 *  Created on: Dec 6, 2016
 *      Author: plong
 */

#ifndef STEREOVISION_HPP_
#define STEREOVISION_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

#include "types.hpp"

#define P_base(imChannels, blockSize) (8 * imChannels * blockSize * blockSize)
#define CALC_DISP(width, scale) (((int) (width * scale / 8) + 15) & -16)

class StereoVision;

typedef void (StereoVision::*compute_func_t)(cv::Mat, cv::Mat, cv::Mat &);

typedef enum {
	stereobm, stereosgbm
} matcher_t;

class StereoVision {
private:
	// Settings
	int blockSize;
	int numDisparities;
	int preFilterCap;

	// Disparity computer
	cv::Ptr<cv::StereoMatcher> left;
	cv::Ptr<cv::StereoMatcher> right;
	cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter;

	compute_func_t compute_disp;
	matcher_t mode;

	// Other
	double avg_time_;

	// Split up constructor
	void loadSettings(void);
	void initMatchers(void);
public:
	StereoVision();
	virtual ~StereoVision();
	void compute_sgbm(cv::Mat frame_l, cv::Mat frame_r, cv::Mat &disp);
	void compute_bm(cv::Mat frame_l, cv::Mat frame_r, cv::Mat &disp);
	void compute(ts_frame<2> &frames, cv::Mat &disp);
	const double avg_time() {
		return avg_time_;
	}
	const int num_disp() {
		return numDisparities;
	}
};

#endif /* STEREOVISION_HPP_ */
