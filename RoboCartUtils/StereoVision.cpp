/*
 * StereoVision.cpp
 *
 *  Created on: Dec 6, 2016
 *      Author: plong
 */

#include "StereoVision.hpp"
#include "settings.hpp"

#define CALL_MEMBER_FN(object,ptrToMember)  ((object)->*(ptrToMember))

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

	double scaleFactor = (double) m["scaleFactor"];

	if (mode == matcher_t::stereosgbm)
		blockSize = 5;
	else if (scaleFactor < 0.99)
		blockSize = 7;
	else
		blockSize = 15;

	int default_width = (int) fs["video"]["src-rez"][0];
	numDisparities = CALC_DISP(default_width, scaleFactor);
	preFilterCap = (int) m["preFilterCap"];
}

void StereoVision::initMatchers(void) {
	if (mode == matcher_t::stereobm) {
		cv::Ptr<cv::StereoBM> matcher = cv::StereoBM::create(numDisparities,
				blockSize);
		matcher->setPreFilterCap(preFilterCap);
		left = matcher;
		compute_disp = &StereoVision::compute_bm;
	} else {
		cv::Ptr<cv::StereoSGBM> matcher = cv::StereoSGBM::create(0,
				numDisparities, blockSize);
		matcher->setPreFilterCap(preFilterCap);
		matcher->setP1(P_base(3, blockSize));
		matcher->setP2(4 * P_base(3, blockSize));
		left = matcher;
		compute_disp = &StereoVision::compute_sgbm;
	}
	filter = cv::ximgproc::createDisparityWLSFilter(left);
	right = cv::ximgproc::createRightMatcher(left);
	filter->setLambda(8000);
	filter->setSigmaColor(1.0);
}

void StereoVision::compute_sgbm(cv::Mat frame_l, cv::Mat frame_r,
		cv::Mat &disp) {
	cv::Mat disp_f;
	cv::cvtColor(frame_l, frame_l, cv::COLOR_BGR2GRAY);
	cv::cvtColor(frame_r, frame_r, cv::COLOR_BGR2GRAY);
	left->compute(frame_l, frame_r, disp_f);
	double minVal;
	double maxVal;
	minMaxLoc(disp_f, &minVal, &maxVal);
	static double factor = 0;
	factor = 63 / (maxVal - minVal) + factor * .75;
	disp_f.convertTo(disp, CV_8U, factor);
}

void StereoVision::compute_bm(cv::Mat frame_l, cv::Mat frame_r, cv::Mat &disp) {
	cv::Mat disp_l, disp_r, disp_f;
	cv::cvtColor(frame_l, frame_l, cv::COLOR_BGR2GRAY);
	cv::cvtColor(frame_r, frame_r, cv::COLOR_BGR2GRAY);
	left->compute(frame_l, frame_r, disp_l);
	right->compute(frame_r, frame_l, disp_r);
	filter->filter(disp_l, frame_l, disp_f, disp_r);
	double minVal;
	double maxVal;
	minMaxLoc(disp_f, &minVal, &maxVal);
	static double factor = 0;
	factor = 63 / (maxVal - minVal) + factor * .75;
	disp_f.convertTo(disp, CV_8U, factor);
}

void StereoVision::compute(ts_frame<2> &frames, cv::Mat &disp) {
	timestamp_t start = clk_t::now();
	(this->*compute_disp)(frames.frame[0], frames.frame[1], disp);
	timestamp_t end = clk_t::now();
	milliseconds diff = duration_cast<milliseconds>(end - start);
	avg_time_ = avg_time_ * AVG_OLD + diff.count() * AVG_NEW;
}
