#ifndef MY_COLOR_SEGMENTATION_H
#define MY_COLOR_SEGMENTATION_H
#include "myColorSegmentationEnum.h"
#include <opencv2\core\core.hpp>
#include <map>
using namespace cv;
using namespace std;



void getLabeledImage(Mat src, Mat* labelImg);
void getColoredLabelMap(Mat labelImg,Mat *labelMap);
int getConnectedComponents(Mat labelImg, Mat *componentImage, map < int, vector<Point> > *componentList);
void getColoredComponents(Mat componentImage, Mat *coloredImage);
void getBoundedBoxImage(Mat originalImage, map < int, vector<Point> > componentList,int threshold, Mat* result);

#endif MY_COLOR_SEGMENTATION_H