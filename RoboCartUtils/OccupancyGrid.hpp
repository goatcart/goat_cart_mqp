#ifndef OCCUPANCYGRID_HPP_
#define OCCUPANCYGRID_HPP_

#include <opencv2/opencv.hpp>

class OccupancyGrid {
private:
	cv::Mat q;
	cv::Size occupancySize;
	double cameraHeight, robotWidth, robotLength, clearance, r, c, deltaN, nt,
			lt, deltaH, wN, wH;
	int count;

public:
	cv::Range xRange, yRange, zRange;
	OccupancyGrid();
	void compute(cv::Mat &disparity, cv::Mat &dispOccupancy, cv::Mat &image3d);
	virtual ~OccupancyGrid();
};

#endif /* OCCUPANCYGRID_HPP_ */
