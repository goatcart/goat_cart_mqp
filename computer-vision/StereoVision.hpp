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

class StereoVision {
private:
	// Settings
    int blockSize;
    int numDisparities;
    int disp12MaxDiff;
    int speckleRange;
    int speckleWindowSize;
    int preFilterCap;
    int uniquenessRatio;

    // Disparity computer
	cv::Ptr<cv::StereoBM> left;
	cv::Ptr<cv::StereoMatcher> right;
	cv::Ptr<cv::ximgproc::DisparityWLSFilter> filter;

	// Other
	double avg_time_;

	// Split up constructor
	void loadSettings(void);
	void initMatchers(void);
public:
	StereoVision();
	virtual ~StereoVision();
	void compute(ts_frame<2> &frames, cv::Mat &disp);
	const double avg_time() { return avg_time_; }
};

#endif /* STEREOVISION_HPP_ */
