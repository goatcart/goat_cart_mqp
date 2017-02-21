/*
 * StereoVision.cpp
 *
 *  Created on: Dec 6, 2016
 *      Author: plong
 */

#include "StereoVision.hpp"
#include "settings.hpp"

using std::chrono::duration_cast;
using std::chrono::milliseconds;

StereoVision::StereoVision() {
	loadSettings();
	initMatchers();
}

StereoVision::~StereoVision() {
	// TODO Auto-generated destructor stub
}

void StereoVision::loadSettings(void) {
	cv::FileStorage fs("params.yml", cv::FileStorage::READ);
	cv::FileNode m = fs["matcher"];

	if ((int) m["mode"] == 1)
		mode = matcher_t::stereosgbm;
	else
		mode = matcher_t::stereobm;

	double scaleFactor = (double) fs["video"]["scale"];

	if (mode == matcher_t::stereosgbm)
		blockSize = 5;
	else if (scaleFactor < 0.99)
		blockSize = 7;
	else
		blockSize = 15;

	int default_width = (int) fs["video"]["src-rez"][0];
	numDisparities = CALC_DISP(default_width, scaleFactor);
	preFilterCap = (int) m["preFilterCap"];
	uniquenessRatio = (int) m["uniquenessRatio"];
	disp12MaxDiff = (int) m["disp12MaxDiff"];
	speckleRange = (int) m["speckleRange"];
	speckleWindowSize = (int) m["speckleWindowSize"];
}

void StereoVision::initMatchers(void) {
	if (mode == matcher_t::stereobm) {
		cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(numDisparities,
				blockSize);
		matcher->setPreFilterCap(preFilterCap);
		matcher->setUniquenessRatio(uniquenessRatio);
		left = matcher;
	} else {
		cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0,
				numDisparities, blockSize);
		matcher->setP1(P_base(3, blockSize));
		matcher->setP2(4 * P_base(3, blockSize));
		matcher->setPreFilterCap(preFilterCap);
		matcher->setUniquenessRatio(uniquenessRatio);
		left = matcher;
	}
	left->setDisp12MaxDiff(disp12MaxDiff);
	left->setSpeckleRange(speckleRange);
	left->setSpeckleWindowSize(speckleWindowSize);
	filter = cv::ximgproc::createDisparityWLSFilter(left);
	right = cv::ximgproc::createRightMatcher(left);
	filter->setLambda(8000);
	filter->setSigmaColor(1.0);
}

void StereoVision::compute(ts_frame<2> &frames, cv::Mat &disp) {
	cv::Mat disp_f;
	double minVal;
	double maxVal;
	static double factor = 0;
	// Compute and scale disparity map
	timestamp_t start = clk_t::now();
	cv::Mat disp_l, disp_r, frame_l, frame_r;
	cv::cvtColor(frames.frame[0], frame_l, cv::COLOR_BGR2GRAY);
	cv::cvtColor(frames.frame[1], frame_r, cv::COLOR_BGR2GRAY);
	left->compute(frame_l, frame_r, disp_l);
	right->compute(frame_r, frame_l, disp_r);
	filter->filter(disp_l, frame_l, disp_f, disp_r);
	minMaxLoc(disp_f, &minVal, &maxVal);
	factor = 63 / (maxVal - minVal) + factor * .75;
	disp_f.convertTo(disp, CV_8U, factor);
	timestamp_t end = clk_t::now();
	// Update average time
	milliseconds diff = duration_cast<milliseconds>(end - start);
	avg_time_ = avg_time_ * AVG_OLD + diff.count() * AVG_NEW;
}
