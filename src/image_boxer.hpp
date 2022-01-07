#ifndef IMAGE_BOXER_HPP
#define IMAGE_BOXER_HPP

#include <iostream>
#include <cmath>

#include "boat_detector_utils.hpp"

#define WEIGHT_X 1.0
#define WEIGHT_Y 1.7
#define DISTANCE_THRESHOLD 2.0
#define NO_CLUSTER -1
#define MIN_CLUSTER_SIZE 10

class ImageBoxer{

public:

	/*
	*@brief constructor
	*@param input_images the set of images where the regions of interest should be detected
	*/
	ImageBoxer(std::vector<cv::Mat> input_images);

	/*
	*@brief extracts the ORB keypoints and features from the images and stores them locally
	*/
	void extractFeatures();

	/*
	*@brief clusters the keypoints found using extractFeatures() and stores the clusters locally
	*/
	void clusterKeypoints();

	/*
	*@brief defines the regions of interest based on the clusters found with clusterKeypoints() and stores them locally.
	*/
	void boxImages();

	/*
	*@brief segments the images and stores them locally
	*N.B. the algorithm is the one based on the distance transform and watershed that was preseted in class
	*It can be found, along the related comments, at https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
	*It does not work well for the images tested
	*/
	void segmentImages();		

	/*
	*@brief shows the detected keypoint in each image
	*/
	void showKeypoints();

	/*
	*@brief shows the detected keypoints and the segmented images
	*/
	void showKeypointSegments();

	/*
	*@brief shows the regions of interest of each image
	*/
	void drawBoxes();

	/*
	*@brief shows the keypoint clusters and the associated regions of interest of each image
	*/
	void drawClusters();

	/*
	*@brief shows each region of interest as a separate image
	*/
	void showWindows();

	/*
	*@brief prints information about each of the detected ORB features
	*/
	void printFeatures();

	/*
	*@brief return the set of feature clusters 
	*@return feature clusters ([image_index][cluster_index][keypoint_index])
	*/
	std::vector< std::vector< std::vector<keypoint_t> > > getClusters();

	/*
	*@brief returns the set of keypoint features
	*@return keypoint features ([global_feature_index])
	*N.B. let 
	*cv::KeyPoint keypoint = *clusters[i][j][k].keypoint_ptr;
	*its features is given by 
	*cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index); 
	*a 1x32 matrix
	*/
	std::vector<cv::Mat> getFeatures();

	/*
	*@breif	returns the set of regions of interest
	*@return regions of interes([image][cluster]->Vec5i(top_left_x, top_left_y, bot_right_x, bot_right_y, cluster_index))
	*N.B. box_coord[i][j] -> i is the image index but j is not necessarily the cluster index (e.g. box_coords[1][0] is the the 0th box of image 1, 
	*and the 0th box could have all the elements from cluster 2 and not cluster 0)
	*/
	std::vector< std::vector<Vec5i> > getCoords();

private:

	// input images
	std::vector<cv::Mat> input_images;

	// vector of keypoints for each image
	std::vector< std::vector<cv::KeyPoint> > keypoints;

	// vector of ORB features for each image
	std::vector<cv::Mat> features;

	// segmented images
	std::vector<cv::Mat> segmented_images;

	// clustered keypoints for each image
	std::vector< std::vector< std::vector<keypoint_t> > > clusters;

	// coordinates of the boxes (top_x, top_y, bot_x, bot_y, cluster_index)
	std::vector< std::vector<Vec5i> > box_coords;
	//N.B. box_coord[i][j] -> i is the image index but j is not necessarily the cluster index (e.g. box_coords[1][0] is the the 0th box of image 1, 
	//and the 0th box could have all the elements from cluster 2 and not cluster 0)

	// set of function used to check local data
	void checkInputImages();
	void checkSegments();
	void checkFeatures();
	void checkClusters();
	void checkBoxes();

};

#endif /*IMAGE_BOXER_HPP*/