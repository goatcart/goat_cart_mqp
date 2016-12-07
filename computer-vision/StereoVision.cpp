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

void StereoVision::loadSettings(void)
{
	cv::FileStorage fs("params.yml", cv::FileStorage::READ);
	cv::FileNode m = fs["matcher"];

	double scaleFactor = (double) m["scaleFactor"];

	if (scaleFactor < 0.99) blockSize = 7;
	else blockSize = 15;

	int default_width = (int) fs["video"]["src-rez"][0];
	numDisparities = CALC_DISP(default_width, scaleFactor);
	disp12MaxDiff = (int) m["disp12MaxDiff"];
	speckleRange = (int) m["speckleRange"];
	speckleWindowSize = (int) m["speckleWindowSize"];
	preFilterCap = (int) m["preFilterCap"];
	uniquenessRatio = (int) m["uniquenessRatio"];
}

void StereoVision::initMatchers(void) {
	left = cv::StereoBM::create(numDisparities, blockSize);
	//left->setDisp12MaxDiff(disp12MaxDiff);
	//left->setSpeckleRange(speckleRange);
	//left->setSpeckleWindowSize(speckleWindowSize);
	left->setPreFilterCap(preFilterCap);
	left->setUniquenessRatio(uniquenessRatio);
    filter = cv::ximgproc::createDisparityWLSFilter(left);
	right = cv::ximgproc::createRightMatcher(left);
	filter->setLambda(8000);
	filter->setSigmaColor(1.0);
}

void StereoVision::compute(ts_frame<2> &frames, cv::Mat &disp)
{
    timestamp_t start = clk_t::now();
	cv::Mat disp_l, disp_r, disp_f;
	left->compute(frames.frame[0], frames.frame[1], disp_l);
	right->compute(frames.frame[1], frames.frame[0], disp_r);
	filter->filter(disp_l, frames.frame[0], disp_f, disp_r);
	cv::ximgproc::getDisparityVis(disp_f, disp, 1.75);
	timestamp_t end = clk_t::now();
    milliseconds diff = duration_cast<milliseconds>(end - start);
    avg_time_ = avg_time_ * AVG_OLD + diff.count() * AVG_NEW;
}
