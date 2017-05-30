#include "myColorSegmentationEnum.h"
#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <map>
#include <cstdio>
using namespace std;
using namespace cv;
/*
Return a matrix with color labels according to given table.
*/
void getLabeledImage(Mat src, Mat* labelImg){
	*labelImg = Mat::zeros(src.rows,src.cols,CV_32FC1);
	Mat hsv;
	cvtColor(src,hsv,CV_BGR2HSV);					//Get HSV image
	int colorHueRanges[13][2] = { { 340, 10 }, { 10, 30 }, { 30, 70 }, { 70, 150 }, { 90, 165 }, { 165, 180 }, { 180, 210 }, { 200, 250 }, { 250, 280 }, { 280, 290 }, { 290, 340 }, { 0, 360 }, {0,360} };
	int colorSatRanges[13][2] = { { 20, 100 }, { 65, 100 }, { 50, 70 }, { 0, 100 }, { 80, 100 }, { 70, 100 }, { 20, 100 }, { 70, 100 }, { 40, 80 }, { 95, 100 }, { 25, 60 }, { 0, 100 }, { 0, 20 } };
	int colorValRanges[13][2] = { { 25, 100 }, { 40, 100 }, { 80, 100 }, { 0, 100 }, { 10, 20 }, { 35, 100 }, { 30, 90 }, { 10, 90 }, { 35, 100 }, { 45, 100 }, { 35, 100 }, { 0, 5 }, {70,100 } };

	int colorCount = 13;
	float hue, sat, val,mySat=0;
	bool colored = false;
	for (int i = 0; i < hsv.rows; i++){
		for (int j = 0; j < hsv.cols; j++)
		{
			colored = false;
			Vec3b curPixel = hsv.at<Vec3b>(i, j);
			hue = 2*curPixel[0];						//OpenCV uses 0-180 for hue values.
			sat = 100 * ((float)curPixel[1] / 255);//OpenCV uses 0-255 values for saturation and value. Table specifies these ranges as 0-100. So we scale.
			val = 100*  ((float)curPixel[2] / 255);

			if ((colorHueRanges[0][0] <= hue || hue <= colorHueRanges[0][1]) &&			//Exception case :RED
				colorSatRanges[0][0] <= sat && sat <= colorSatRanges[0][1] &&
				colorValRanges[0][0] <= val && val <= colorValRanges[0][1])
				{

					labelImg->at<int>(i, j) = 0;
					colored = true;

				}


			for (int k = 1; k < colorCount; k++)
			{
	
				if (colorHueRanges[k][0] <= hue && hue <= colorHueRanges[k][1] &&			//Check the color 
					colorSatRanges[k][0] <= sat && sat <= colorSatRanges[k][1] &&
					colorValRanges[k][0] <= val && val <= colorValRanges[k][1])
					{

						labelImg->at<int>(i,j) = k;
						colored = true;
	
					}
			}
			if (!colored)
				labelImg->at<int>(i, j) = MYCOLOR_OTHERS;									//If there is no color detected, mark it as "other"
		}
	}
}
/*
Visualize the color labels by assigning a color to them.
*/
void getColoredLabelMap(Mat labelImg, Mat *labelMap)
{
	//The colors representing respective labels
	Vec3b colorRepresents[14] = { Vec3b(255, 0, 0), Vec3b(255, 165, 0), Vec3b(255, 255, 0), Vec3b(127, 255, 0), Vec3b(0, 100, 0), Vec3b(0, 255, 255), Vec3b(173, 216, 230), Vec3b(0, 0, 128), Vec3b(238, 130, 238), Vec3b(160, 32, 240), Vec3b(255, 192, 203), Vec3b(0, 0, 0), Vec3b(255, 255, 255), Vec3b(190, 190, 190) };
	*labelMap = Mat::zeros(labelImg.rows, labelImg.cols, CV_8UC3);
	for (int i = 0; i < labelImg.rows; i++){
		for (int j = 0; j < labelImg.cols; j++)
		{
			labelMap->at <Vec3b>(i, j) = colorRepresents[labelImg.at<int>(i, j)];
		}
	}
	cvtColor(*labelMap,*labelMap,CV_RGB2BGR);		//Convert to BGR
}
/*
Simple union-find for disjoint sets with path compression.
Used in component analysis to keep track of equilavencies.
*/
class UnionFind{
public:
	vector<int> parent;
	int unionCount = 0;
	UnionFind(){}
	/*Find the equilavent label*/
	int find(int X)
	{
		int curX = X;

		while (parent[curX] != -1) { curX = parent[curX]; }
		if(X!=curX)parent[X] = curX;
		return curX;
	}
	/*Unify labels*/
	void set(int X,int Y){
		int curX = find(X);
		int curY = find(Y);
		if (curX != curY)
		{
			parent[curY] = curX;
			unionCount++;
		}
	}
	/*Extend the parent vector*/
	void update()
	{ 
		parent.push_back(-1); 
	}
	/*Returns the number of labels processed so far */
	int getUncompressedSize()
	{
		return (int)parent.size();
	}
	/*Returns the number of disjoint labels/sets */
	int getDisjointSetCount()
	{
		return (int)parent.size()-unionCount;
	}
};
/*
Returns the number of connected components.
Also stores the component information in componentList.
Also returns the labeled image matrix.
*/
int getConnectedComponents(Mat labelImg, Mat *componentImage,map < int,vector<Point> > *componentList){
	*componentImage = Mat::zeros(labelImg.rows, labelImg.cols, CV_32FC1);
	int lastLabel = 0,candidateLabel;
	UnionFind unionFind;
	for (int i = 0; i < labelImg.rows; i++){
		for (int j = 0; j < labelImg.cols; j++)
		{
			if ((i>0 && j>0) && labelImg.at<int>(i, j) == labelImg.at<int>(i - 1, j) && labelImg.at<int>(i, j) == labelImg.at<int>(i, j - 1)){	///Check if both neighbors are same
				candidateLabel = min(componentImage->at<int>(i - 1, j), componentImage->at<int>(i, j - 1));									//Get the smaller label
				unionFind.set(componentImage->at<int>(i - 1, j), componentImage->at<int>(i, j - 1));										//If so unify their labels
				
			}
			else if (i>0 && labelImg.at<int>(i, j) == labelImg.at<int>(i - 1, j))						//Check north neighbor
			{
				candidateLabel = componentImage->at<int>(i - 1, j);
			}
			else if (j>0 && labelImg.at<int>(i, j) == labelImg.at<int>(i, j - 1))						//Check west neighbor
			{
				candidateLabel = componentImage->at<int>(i, j - 1);
			}
			else{
				candidateLabel = lastLabel; lastLabel++;												//Else a new label is created
				unionFind.update();
			}
			componentImage->at<int>(i, j) = candidateLabel;
		}
	}
	for (int i = 0; i < labelImg.rows; i++){
		for (int j = 0; j < labelImg.cols; j++)
		{
			componentImage->at<int>(i, j) = unionFind.find(componentImage->at<int>(i, j));				//Set the labels to their respective equilavents
			if (componentList->find(componentImage->at<int>(i, j)) == componentList->end())				//If this label is not seen before(for listing purposes) 
				componentList->insert(pair<int, vector<Point> >(componentImage->at<int>(i, j),vector<Point>()));	//Create a new list for this label
			componentList->at(componentImage->at<int>(i, j)).push_back(Point(j,i));							//Push the pixel location into this label's list
		}
	}
	return  unionFind.getDisjointSetCount();
}


/*
Visualize the connected components
*/
void getColoredComponents(Mat componentImage,Mat* coloredImage)
{
	RNG rng(247); Vec3b color;
	*coloredImage = Mat::zeros(componentImage.rows, componentImage.cols, CV_8UC3);
	map<int,Vec3b> blobColors;														//List of which label assigned to which color

	for (int i = 0; i < componentImage.rows; i++){
		for (int j = 0; j < componentImage.cols; j++)
		{
			if (blobColors.find(componentImage.at<int>(i, j)) == blobColors.end())		//If this label is not seen before in color list
			{
				 color = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));	//Create a random color
				blobColors.insert(pair<int, Vec3b>(componentImage.at<int>(i, j),color));		//Add this color to color list
			}
			coloredImage->at<Vec3b>(i, j) = blobColors[componentImage.at<int>(i, j)];		//Color according to list
		}
	}
}

void getBoundedBoxImage(Mat originalImage,map < int, vector<Point> > componentList,int threshold,Mat* result)
{
	cout << "Component stats for threshold " << threshold << ":" << endl;

	*result = originalImage.clone();
	RNG rng(715);
	vector<int> foundSizes;
	for (map<int, vector<Point> >::iterator it = componentList.begin(); it != componentList.end(); ++it)			//Traverse the component list
	{
		if (it->second.size() < threshold) continue;																//If lower than threshold,continue
		foundSizes.push_back(it->second.size());																	//Store the size of the component
		Rect box=boundingRect(it->second);																			//Get the bounding box of the component
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));						//Draw bounding box with random color
		rectangle(*result, box.tl(),box.br(), color, 2);
		cout << "Centroid: " << "[" << (box.tl() + box.br()).x / 2 << "," << (box.tl() + box.br()).y / 2 << "];" << " Size: " << it->second.size() << endl;	//Print center and size
	}

	/*Calculate statistics*/
	int total = 0;
	for (int i = 0; i < foundSizes.size(); i++)
	{
		total += foundSizes[i];
	}
	float mean = (float)total / foundSizes.size();
	float variance = 0;
	for (int i = 0; i < foundSizes.size(); i++)
	{
		variance += (foundSizes[i] - mean)*(foundSizes[i] - mean);
	}
	variance = sqrt(variance/foundSizes.size());
	cout << "Count: " << foundSizes.size() << endl;
	cout << "Mean: " << mean << endl; 
	cout << "Variance: " << variance << endl<<endl;
}