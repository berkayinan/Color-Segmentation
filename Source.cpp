#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include <iostream>
#include "myColorSegmentation.h"
#include <map>
using namespace std;
using namespace cv;


int main()
{
	Mat img = imread("baseball.jpg");				//Read Image
	Mat *labeledImage = new Mat();
	Mat *labelMap = new Mat();
	int blurMaskSize = 3; GaussianBlur(img, img, Size(blurMaskSize, blurMaskSize), 3);			//Optional Blur
	getLabeledImage(img, labeledImage);																//Get color labels
	getColoredLabelMap(*labeledImage,labelMap);														//Visualize color labels
	imshow("Labeled Image", *labelMap); imwrite("colorLabelGuitar.jpg", *labelMap);
	Mat componentImage, coloredComponentImage;														//Connected component labels and visualized version
	map<int, vector <Point> > componentList;														//List of connected components

	int labelCount = getConnectedComponents(*labeledImage, &componentImage, &componentList);
	cout << "NUMBER OF CONNECTED COMPONENTS: "<<labelCount << endl;
	getColoredComponents(componentImage, &coloredComponentImage);
	imshow("Connected Components", coloredComponentImage); imwrite("componentsGuitar.jpg", coloredComponentImage);


	Mat boundedBoxImage;																			//Show bounding boxes with threshold
	getBoundedBoxImage(img, componentList, 300, &boundedBoxImage);
	imshow("Bounded Box Image 300", boundedBoxImage); imwrite("boundingBoxGuitar300.jpg", boundedBoxImage);
	getBoundedBoxImage(img, componentList, 1000, &boundedBoxImage);
	imshow("Bounded Box Image 1000", boundedBoxImage); imwrite("boundingBoxGuitar1000.jpg", boundedBoxImage);
	getBoundedBoxImage(img, componentList, 10000, &boundedBoxImage);
	imshow("Bounded Box Image 10000", boundedBoxImage); imwrite("boundingBoxGuitar10000.jpg", boundedBoxImage);
	waitKey(0);
	return 0;
}