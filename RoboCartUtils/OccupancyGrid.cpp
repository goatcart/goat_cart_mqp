#include "OccupancyGrid.hpp"

/*
 OccupancyGrid::OccupancyGrid(cv::Mat q, cv::Size occupancySize,
 double cameraHeight, double robotWidth, double robotLength, double clearance,
 cv::Range xRange, cv::Range yRange, cv::Range zRange,
 double r, double c, double deltaN, double nt, double lt, double deltaH, double wN, double wH);
 */

OccupancyGrid::OccupancyGrid() {
	cv::FileStorage fs("params.yml", cv::FileStorage::READ);
	cv::FileStorage ext("extrinsics.yml", cv::FileStorage::READ);
	cv::FileNode og = fs["og"];

	og["occupancySize"] >> occupancySize;
	cameraHeight = (double) og["cameraHeight"];
	robotWidth = (double) og["robotWidth"];
	robotLength = (double) og["robotLength"];
	clearance = (double) og["clearance"];
	og["xRange"] >> xRange;
	og["yRange"] >> yRange;
	og["zRange"] >> zRange;
	ext["Q"] >> q;
	r = (double) og["r"];
	c = (double) og["c"];
	deltaN = (double) og["deltaN"];
	nt = (double) og["nt"];
	lt = (double) og["lt"];
	deltaH = (double) og["deltaH"];
	wN = (double) wN;
	wH = (double) wH;
	count = 0;
}

void OccupancyGrid::compute(cv::Mat &disparity, cv::Mat &dispOccupancy,
		cv::Mat &image3d) {
	cv::reprojectImageTo3D(disparity, image3d, q, true);
	cv::Mat occupancy(occupancySize.height, occupancySize.width, CV_32FC1);
	occupancy = cv::Scalar(0);
	cv::Mat height(occupancySize.height, occupancySize.width, CV_32FC1);
	height = cv::Scalar(0);

	for (int i = 0; i < image3d.rows; i++) {
		for (int j = 0; j < image3d.cols; j++) {
			cv::Point3f pt = image3d.at<cv::Point3f>(i, j);
			double h = cameraHeight - pt.y;
			if (pt.x > xRange.end || pt.x < xRange.start || h > yRange.end
					|| h < yRange.start || pt.z > zRange.end
					|| pt.z < zRange.start) {
				continue;
			}

			float scaledZ = (pt.z - zRange.start) / (zRange.end - zRange.start);
			float scaledX = (pt.x - xRange.start) / (xRange.end - xRange.start);

			int row = occupancySize.height
					- floor(scaledZ * occupancySize.height);
			int col = floor(scaledX * occupancySize.width);
			occupancy.at<float>(row, col) += 1;
			height.at<float>(row, col) += h;
		}
	}

	cv::Mat test;
	normalize(occupancy, test, 0, 255, CV_MINMAX, CV_8UC1);

	dispOccupancy.create(occupancy.rows, occupancy.cols, CV_8UC3);
	dispOccupancy = cv::Scalar(0, 0, 0); //TODO: remove this, not needed.
	float xCam = occupancySize.width / 2, yCam = occupancySize.height;

	for (int i = 0; i < occupancy.rows; i++) {
		for (int j = 0; j < occupancy.cols; j++) {
			if (occupancy.at<float>(i, j) > 0) {
				float xPt = j, yPt = i;
				float distanceToCamera = sqrt(
						(xCam - xPt) * (xCam - xPt)
								+ (yCam - yPt) * (yCam - yPt));
				//cout << distanceToCamera << endl;
				float adjustedNum = occupancy.at<float>(i, j) * r
						/ (1 + exp(-distanceToCamera * c));
				float pijNum = 1 - exp(-(adjustedNum / deltaN));
				float lijNum = pijNum != 1 ? log(pijNum / (1 - pijNum)) : 0;

				float avgHeight =
						occupancy.at<float>(i, j) > 0 ?
								height.at<float>(i, j)
										/ occupancy.at<float>(i, j) :
								0;
				float pijHeight = 1 - exp(-avgHeight / deltaH);
				float lijHeight =
						pijHeight != 1 ? log(pijHeight / (1 - pijHeight)) : 0;

				float avgProb = wN * lijNum + wH * lijHeight;
				//cout << "occ " << occupancy.at<float>(i,j) << " adjNum " << adjustedNum << " adj/deltan " << adjustedNum/deltaN << " expadjdeltan " << exp(-adjustedNum/deltaN) << " pijnum " << pijNum << " lijnum " << lijNum << " avgheight "
				//		<< avgHeight << " avgheightdeltah " << avgHeight/deltaH << " expavghdeltah " << exp(-avgHeight/deltaH) << " pijheight " << pijHeight << " lijheight " << lijHeight << " avgprob " << avgProb << " deltah " << deltaH << " deltaN " << deltaN  << endl;
				//cout << occupancy.at<float>(i, j) << "|" << pijNum << "|" << lijNum << endl;
				if (avgProb < nt) {
					dispOccupancy.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
				} else if (lijNum >= lt) {
					dispOccupancy.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);
				} else {
					dispOccupancy.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);
				}
			} else {
				dispOccupancy.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
		}
	}

	cv::Mat mask;
	cv::inRange(dispOccupancy, cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),
			mask);
	dispOccupancy.setTo(cv::Scalar(0, 0, 0), mask);

	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::morphologyEx(dispOccupancy, dispOccupancy, cv::MORPH_OPEN, kernel);

	double cellWidth = (xRange.end - xRange.start) / occupancy.cols;
	int dilationN = (robotWidth / 2 + clearance) / cellWidth;

	cv::Mat dilationKernel = cv::getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * dilationN + 1, 1));
	cv::dilate(dispOccupancy, dispOccupancy, dilationKernel);
}

OccupancyGrid::~OccupancyGrid() {
	// TODO Auto-generated destructor stub
}

