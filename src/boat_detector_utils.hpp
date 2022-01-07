#ifndef BOAT_DETECTOR_UTILS_HPP
#define BOAT_DETECTOR_UTILS_HPP

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

#define TOP_X 0
#define TOP_Y 1
#define BOT_X 2
#define BOT_Y 3
#define CLUSTER_INDEX 4
#define BOX_BORDER 5
#define BOX_COLOR cv::Scalar(50,100,200)

//typical use Vec5i(top_left_x, top_left_y, bot_right_x, bot_right_y, cluster_index)
typedef cv::Vec<int, 5> Vec5i;

typedef struct{
	cv::KeyPoint* keypoint_ptr;
	int keypoint_index;
} keypoint_t;

class BoatDetectorUtils{

public:

	/*
	*@brief loads a set of images from a directory
	*@param path the directory path where the images are located
	*/
	static std::vector<cv::Mat> loadImages(cv::String path){
		
		std::cout << "Loading images..." << std::endl;

		std::vector<cv::String> filenames;
		std::vector<cv::Mat> input_images;

		cv::glob(path, filenames, false);
		for (size_t i = 0; i < filenames.size(); ++i){
		cv::Mat img = cv::imread(filenames[i]);
		//resize(img, img, Size(img.cols / 2.0, img.rows / 2.0));
		input_images.push_back(img);

		std::cout << "Loaded image " << i << std::endl;
		}
		std::cout << "Finished loading images...\n" << std::endl;
		return input_images;
	}

	/*
	*@brief displays a set of images on the screen
	*@param input_images the images to be shown
	*/
	static void showImages(std::vector<cv::Mat> input_images){

		if (input_images.size() < 1){
			std::cout << "No images to show\n" << std::endl;
			return;
		}

		for (size_t i = 0; i < input_images.size(); ++i){
			cv::namedWindow("Input images");
			cv::imshow("Input images", input_images[i]);
			cv::waitKey(0);
		}
		cv::destroyAllWindows();
	} 

	/*
	*@brief function that displays the regions of interest and prints information about the keypoints
	*@param	images set of images where the regions of interest have been located
	*@param clusters clustered keypoints of the images ([image][cluster][keypoint])
	*@param features matrix of ORB features for each image ([image])
	*@param boxes the set of regions of interest ([image][cluster])
	*@param window_index used to keep track of how many regions of interest have already been classified
	*/
	static void test_boxer(std::vector<cv::Mat> images, std::vector< std::vector< std::vector<keypoint_t> > > clusters, 
	std::vector<cv::Mat> features, std::vector< std::vector<Vec5i> > boxes, int window_index){

		for (size_t i = 0; i < boxes.size(); i++)	//iterate ofver images
			for (size_t j = 0; j < boxes[i].size(); j++){	//iterate over boxes
				
				//cosntruct a window representing the region of interest
				int x = boxes[i][j][TOP_X];
				int y = boxes[i][j][TOP_Y];
				int width = boxes[i][j][BOT_X] - boxes[i][j][TOP_X];
				int height = boxes[i][j][BOT_Y] - boxes[i][j][TOP_Y];
				int c_index = boxes[i][j][CLUSTER_INDEX];
				cv::Mat window = images[i](cv::Rect(x, y, width, height));

				//check data consistency
				if (c_index != static_cast<int>(j))
					std::cout << "\n\n\n!!!NON MATCHING INDEXES!!!(c_index = " + std::to_string(c_index) + 
					", j = " + std::to_string(static_cast<int>(j)) + ")" + "\nx = " + std::to_string(x) + 
					", y = " + std::to_string(y) + ", width = " + std::to_string(width) + ", height = " + 
					std::to_string(height) + "\n\n" << std::endl;

				//show region of interest
				cv::String window_name = "Box " + std::to_string(j) + " associated with cluster " + 
				std::to_string(c_index) + " of image " + std::to_string(i);
				cv::namedWindow(window_name);
				cv::imshow(window_name, window);

				for (size_t k = 0; k < clusters[i][c_index].size(); k++){	//iterate over keypoints in cluster/region of interest
					cv::KeyPoint keypoint = *clusters[i][c_index][k].keypoint_ptr;	//keypoint in the cluster 
					
					//check data consistency
					if (keypoint.class_id != c_index)
						std::cout << "\n\n\n!!!NON MATCHING CLUSTERS!!!(c_index = " + std::to_string(c_index) + 
						", class_id = " + std::to_string(keypoint.class_id) + ")\n\n" << std::endl;
					
					//print information about keypoint
					std::cout << "--KeyPoint of index " + std::to_string(k) + " from cluster " +
					std::to_string(c_index) + " and class_id (effective cluster) " + 
					std::to_string(keypoint.class_id) + ", window " + std::to_string(j) + ", image " + 
					std::to_string(i) + " at coordinates (" + std::to_string(keypoint.pt.x) + ", " + 
					std::to_string(keypoint.pt.y) + "):" << std::endl;

					//extract and print the vectorr of ORB features of the keypoint
					cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
					std::cout << keypoint_features << std::endl;
				}
				std::cout << "WINDOW_INDEX: " << window_index++ << std::endl;
				std::cout << "\n\n\n";
				cv::waitKey(0);
				cv::destroyAllWindows();
			}
	}

	/*
	*@brief function that displays the regions of interest, the label associated with said region, and prints information about the keypoints
	*@param	images set of images where the regions of interest have been located
	*@param clusters clustered keypoints of the images ([image][cluster][keypoint])
	*@param features matrix of ORB features for each image ([image])
	*@param boxes the set of regions of interest ([image][cluster])
	*@param labels the set of labels associated with each region of interest ([cluster])
	*/
	static void test_boxer(std::vector<cv::Mat> images, std::vector< std::vector< std::vector<keypoint_t> > > clusters, 
		std::vector<cv::Mat> features, std::vector< std::vector<Vec5i> > boxes, std::vector<int> labels){

		int b_index, b_val;	//binary lable index(region index) and value
		b_index = 1;

		for (size_t i = 0; i < boxes.size(); i++)	//iterate over images
			for (size_t j = 0; j < boxes[i].size(); j++){	//iterate over boxes
				
				b_val = labels[b_index-1];
				
				//cosntruct a window representing the region of interest
				int x = boxes[i][j][TOP_X];
				int y = boxes[i][j][TOP_Y];
				int width = boxes[i][j][BOT_X] - boxes[i][j][TOP_X];
				int height = boxes[i][j][BOT_Y] - boxes[i][j][TOP_Y];
				int c_index = boxes[i][j][CLUSTER_INDEX];
				cv::Mat window = images[i](cv::Rect(x, y, width, height));

				//check data consistency
				if (c_index != static_cast<int>(j))
					std::cout << "\n\n\n!!!NON MATCHING INDEXES!!!(c_index = " + std::to_string(c_index) + 
					", j = " + std::to_string(static_cast<int>(j)) + ")" + "\nx = " + std::to_string(x) + 
					", y = " + std::to_string(y) + ", width = " + std::to_string(width) + ", height = " + 
					std::to_string(height) + "\n\n" << std::endl;

				//select the regions of interest to display
				//condition: b_val >= 0 -> display all the regions
				//condition: b_val == 1 -> display only regions where boats have been located
				//condition: b_val == 0 -> display only regions where boats have not been located
				if (b_val >= 0){	

					//show region of interest
					cv::String window_name = "Box " + std::to_string(j) + " associated with cluster " + 
					std::to_string(c_index) + " of image " + std::to_string(i);
					cv::namedWindow(window_name);
					cv::imshow(window_name, window);

					for (size_t k = 0; k < clusters[i][c_index].size(); k++){	//iterate over keypoints in cluster/region of interest
						cv::KeyPoint keypoint = *clusters[i][c_index][k].keypoint_ptr;	//keypoint in the cluster 
						
						//check data consistency
						if (keypoint.class_id != c_index)
							std::cout << "\n\n\n!!!NON MATCHING CLUSTERS!!!(c_index = " + std::to_string(c_index) + 
							", class_id = " + std::to_string(keypoint.class_id) + ")\n\n" << std::endl;

						//print information about keypoint
						std::cout << "--KeyPoint of index " + std::to_string(k) + " from cluster " +
						std::to_string(c_index) + " and class_id (effective cluster) " + 
						std::to_string(keypoint.class_id) + ", window " + std::to_string(j) + ", image " + 
						std::to_string(i) + " at coordinates (" + std::to_string(keypoint.pt.x) + ", " + 
						std::to_string(keypoint.pt.y) + "):" << std::endl;

						//extract and print the vectorr of ORB features of the keypoint
						cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
						std::cout << keypoint_features << std::endl;
					}
					std::cout << "B_INDEX: " << b_index << std::endl;
					std::cout << "B_VAL: " << b_val << std::endl;
					std::cout << "\n\n\n";
					cv::waitKey(0);
					cv::destroyAllWindows();
				}
				b_index++;
			}
	}

};

#endif /*BOAT_DETECTOR_UTILS_HPP*/