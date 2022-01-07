#include "image_boxer.hpp"

	
ImageBoxer::ImageBoxer(std::vector<cv::Mat> input_images){
	for (size_t i = 0; i < input_images.size(); i++)
		this->input_images.push_back(input_images[i]);
}

void ImageBoxer::extractFeatures(){

	std::cout << "Extracting features from images..." << std::endl;

	for (size_t i = 0; i < input_images.size(); ++i){	//iterate over images
			
		cv::Ptr<cv::ORB> detector = cv::ORB::create();
		std::vector<cv::KeyPoint> image_keypoints;	//set  of image keypoint
		cv::Mat image_features;	//set of image features

		detector -> detectAndCompute(input_images[i], cv::noArray(), image_keypoints, image_features);
		keypoints.push_back(image_keypoints);
		features.push_back(image_features);

		std::cout << "Extracted features from image " << i << std::endl;
	}
	std::cout << "Finished extracting features from images...\n" << std::endl;
}

void ImageBoxer::clusterKeypoints(){

	std::cout << "Clustering keypoints..." << std::endl;

	for (size_t i = 0; i < input_images.size(); i++){	//iterate over images
			
		std::cout << "Clustering keypoints of image " << i << std::endl;
			
		int cluster_index = 0;	//indicates the name of the cluster
		bool finished_clustering = false;
		
		std::vector<keypoint_t> keypoints2cluster;	//keypoints of the image that are to be clustered
		std::vector<keypoint_t> cluster_candidates;	//keypoints of the image that might belong to a certain cluster
		std::vector< std::vector<keypoint_t> > image_clusters;	//keypoint of the image that do belong to a certain cluster 
			
		//iterate over all the keypoints in an image and set them as points that are to be clustered
		for (size_t j = 0; j < keypoints[i].size(); j++)	
			keypoints2cluster.push_back(keypoint_t{.keypoint_ptr = &keypoints[i][j], 
				.keypoint_index = static_cast<int>(j)});

		while (!finished_clustering){
				
			//there are no more points to be clustered
			if (keypoints2cluster.size() < 1){
				finished_clustering = true;
				continue;
			}

			//a random point from which the linkage-based clustering can grow
			keypoint_t seed = keypoints2cluster.back();
			keypoints2cluster.pop_back();

			//create new cluster and initialize the seed
			seed.keypoint_ptr->class_id = cluster_index;
			cluster_candidates.push_back(seed);

			//keypoints that belong to the newly created cluster
			std::vector<keypoint_t> keypoint_cluster;	

			while (cluster_candidates.size() > 0){	//while there are still seeds
					
				//check if the keypoints that have not yet been clustered belong to the new cluster
				for (size_t j = 0; j < keypoints2cluster.size(); ){

					//check if the distance is within the threshold
					if ((WEIGHT_X*std::abs(seed.keypoint_ptr->pt.x - keypoints2cluster[j].keypoint_ptr->pt.x) +
						 WEIGHT_Y*std::abs(seed.keypoint_ptr->pt.y - keypoints2cluster[j].keypoint_ptr->pt.y)) < 
						 DISTANCE_THRESHOLD*(seed.keypoint_ptr->size)){
								
							//if so add the keypoint to be a cluster seed
							keypoint_t new_member = keypoints2cluster[j];
							keypoints2cluster.erase(keypoints2cluster.begin()+j);
							new_member.keypoint_ptr->class_id = cluster_index;
							cluster_candidates.push_back(new_member);
					}
					else//otherwise go to the next keypoint that has not been clustered
						j++;
				}
				//add the seed to the new cluster and select new seed
				keypoint_cluster.push_back(seed);
				cluster_candidates.erase(cluster_candidates.begin());
				seed = cluster_candidates[0];
			}
			//check if the new cluster reaches the minimum allowed size
			if (keypoint_cluster.size() >= MIN_CLUSTER_SIZE){
				image_clusters.push_back(keypoint_cluster);
				cluster_index++;
			}
			else
				for (size_t k = 0; k < keypoint_cluster.size(); k++)
					keypoint_cluster[k].keypoint_ptr->class_id = NO_CLUSTER;
				
		}
		clusters.push_back(image_clusters);
	}
	std::cout << "Finished clustering keypoints...\n" << std::endl;
}

void ImageBoxer::boxImages(){

	std::cout << "Boxing keypoints..." << std::endl;

	for (size_t i = 0; i < input_images.size(); i++){	//iterate over images
			
		std::cout << "Boxing keypoints of image " << i << std::endl;
			
		int cluster_index;	//index of the cluster associated with the region of interest
		bool finished_boxing = false;

		std::vector<cv::KeyPoint*> keypoints2analyze;	//keypoints in the images that need to be analyzed to determine the regions of interst
		std::vector<Vec5i> image_boxes;	//all the regions detected in the image
		int top_x, bot_x, top_y, bot_y;	//coordinates of the diametrically opposed corners (top-left and bottom-right) of a region of interest 

		//analyze all the keypoints in the image
		for (size_t j = 0; j < keypoints[i].size(); j++)
			keypoints2analyze.push_back(&keypoints[i][j]);

		while (!finished_boxing){
				
			//there are no more keypoints to analyze
			if (keypoints2analyze.size() < 1){
				finished_boxing = true;
				continue;
			}

			//extract a random seed keypoint so that the cluster associated with it will be analyzed 
			cv::KeyPoint* starting_point = keypoints2analyze.back();
			keypoints2analyze.pop_back();

			cluster_index = starting_point->class_id;	//index of the detected region
			top_x = bot_x = starting_point->pt.x;	//initialize the coordiantes of the region
			top_y = bot_y = starting_point->pt.y;	//initialize the coordiantes of the region

			for (size_t j = 0; j < keypoints2analyze.size(); ){	//iterate over the keypoints that have not  yet been analyzed
					
				//keypoint does not belong to a cluster so it does not need to be analyzed
				if (keypoints2analyze[j]->class_id == NO_CLUSTER){
					keypoints2analyze.erase(keypoints2analyze.begin()+j);
					continue;
				}
					
				//analyze all the keypoints belonging to the same cluster
				if (keypoints2analyze[j]->class_id == cluster_index){

					if (keypoints2analyze[j]->pt.x < top_x)	//left-most keypoint found so far
						top_x = keypoints2analyze[j]->pt.x;
					if (keypoints2analyze[j]->pt.x > bot_x)	//right-most heypoint found so far
						bot_x = keypoints2analyze[j]->pt.x;
					if (keypoints2analyze[j]->pt.y < top_y)	//up-most keypoint found so far
						top_y = keypoints2analyze[j]->pt.y;
					if (keypoints2analyze[j]->pt.y > bot_y)	//down-most keypoint found so far
						bot_y = keypoints2analyze[j]->pt.y;

					keypoints2analyze.erase(keypoints2analyze.begin()+j);
				}
				else
					j++;
			}
			//check if the seed keypoint was associated with a cluster, if so add the related region of interest
			if (cluster_index != NO_CLUSTER)
				image_boxes.push_back(Vec5i(top_x, top_y, bot_x, bot_y, cluster_index));
		}
		box_coords.push_back(image_boxes);
	}
	std::cout << "Finished boxing keypoints...\n" << std::endl;	
}

//N.B. the algorithm is the one based on the distance transform and watershed that was preseted in class
//It can be found, along the related comments, at https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
//It does not work well for the images tested
void ImageBoxer::segmentImages(){

	std::cout << "Segmenting images..." << std::endl;

	cv::Mat kernel = (cv::Mat_<float>(3,3) << 
			  	    1,  1, 1,
                    1, -8, 1,
                    1,  1, 1);

	for (size_t i = 0; i < input_images.size(); ++i){
		cv::Mat imgLaplacian;
   		cv::filter2D(input_images[i], imgLaplacian, CV_32F, kernel);
   		cv::Mat sharp;
   		input_images[i].convertTo(sharp, CV_32F);
   		cv::Mat imgResult = sharp - imgLaplacian;
    
   		imgResult.convertTo(imgResult, CV_8UC3);
   		imgLaplacian.convertTo(imgLaplacian, CV_8UC3);

   		cv::Mat bw;
   		cv::cvtColor(imgResult, bw, cv::COLOR_BGR2GRAY);
   		cv::threshold(bw, bw, 40, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

   		cv::Mat dist;
   		cv::distanceTransform(bw, dist, cv::DIST_L2, 3);
   		cv::normalize(dist, dist, 0, 1.0, cv::NORM_MINMAX);

   		cv::threshold(dist, dist, 0.4, 1.0, cv::THRESH_BINARY);
   		cv::Mat kernel1 = cv::Mat::ones(3, 3, CV_8U);
   		cv::dilate(dist, dist, kernel1);

   		cv::Mat dist_8u;
   		dist.convertTo(dist_8u, CV_8U);
   		std::vector<std::vector<cv::Point> > contours;
   		cv::findContours(dist_8u, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		cv::Mat markers = cv::Mat::zeros(dist.size(), CV_32S);
   		for (size_t j = 0; j < contours.size(); j++)
       		cv::drawContours(markers, contours, static_cast<int>(j), 
       			cv::Scalar(static_cast<int>(j)+1), -1);
   		cv::circle(markers, cv::Point(5,5), 3, cv::Scalar(255), -1);
   		cv::Mat markers8u;
   		markers.convertTo(markers8u, CV_8U, 10);

   		cv::watershed(imgResult, markers);
   		cv::Mat mark;
   		markers.convertTo(mark, CV_8U);
   		cv::bitwise_not(mark, mark);
   		std::vector<cv::Vec3b> colors;
   		for (size_t j = 0; j < contours.size(); j++)
   		{
       		int b = cv::theRNG().uniform(0, 256);
       		int g = cv::theRNG().uniform(0, 256);
       		int r = cv::theRNG().uniform(0, 256);
       		colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
   		}
   		cv::Mat dst = cv::Mat::zeros(markers.size(), CV_8UC3);
   		for (int j = 0; j < markers.rows; j++)
   			for (int k = 0; k < markers.cols; k++)
       		{
           		int index = markers.at<int>(j,k);
           		if (index > 0 && index <= static_cast<int>(contours.size()))
               		dst.at<cv::Vec3b>(j,k) = colors[index-1];
           	}
        segmented_images.push_back(dst);
        std::cout << "Segmented image " << i << std::endl;
    }
    std::cout << "Finished segmenting images...\n" << std::endl;
}	

void ImageBoxer::checkInputImages(){
	if (ImageBoxer::input_images.size() < 1){
		std::cout << "No images to show yet" << std::endl;
		return;
	}
}

void ImageBoxer::checkSegments(){
	if (ImageBoxer::segmented_images.size() < 1){
		std::cout << "No segmented images to show" << std::endl;
		return;
	}
}

void ImageBoxer::checkFeatures(){
	if (ImageBoxer::features.size() < 1){
		std::cout << "No keypoints detected yet" << std::endl;
		return;
	}
}

void ImageBoxer::checkClusters(){
	if (ImageBoxer::clusters.size() < 1){
		std::cout << "No keypoints clustered yet" << std::endl;
		return;
	}
}

void ImageBoxer::checkBoxes(){
	if (ImageBoxer::box_coords.size() < 1){
		std::cout << "No boxes computed yet" << std::endl;
		return;
	}
}	

void ImageBoxer::showKeypoints(){

	//check state of local data
	checkInputImages();
	checkFeatures();

	for (size_t i = 0; i < input_images.size(); ++i){	//iterate over the images
		cv::Mat image_keypoints;
		cv::drawKeypoints(input_images[i], keypoints[i], image_keypoints);
		cv::namedWindow("Keypoints of input images");
		cv::imshow("Keypoints of input images", image_keypoints);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();	
}

void ImageBoxer::showKeypointSegments(){

	//check state of local data
	checkSegments();
	checkFeatures();

	for (size_t i = 0; i < input_images.size(); i++){	//iterate over the segmented images
		cv::Mat image_keypoint_segments;
		cv::drawKeypoints(segmented_images[i], keypoints[i], image_keypoint_segments);
		cv::namedWindow("Keypoints and segments of input images");
		cv::imshow("Keypoints and segments of input images", image_keypoint_segments);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();
}

void ImageBoxer::drawBoxes(){

	//check state of local data
	checkInputImages();
	checkFeatures();
	checkClusters();
	checkBoxes();

	for (size_t i = 0; i < input_images.size(); i++){	//iterate over images
		cv::Mat boxed_image = input_images[i];
		for (size_t j = 0; j < box_coords[i].size(); j++){	//iterated over regions of interest
			cv::rectangle(boxed_image, 
				cv::Point(box_coords[i][j][TOP_X]-BOX_BORDER, box_coords[i][j][TOP_Y]-BOX_BORDER), 
				cv::Point(box_coords[i][j][BOT_X]+BOX_BORDER, box_coords[i][j][BOT_Y]+BOX_BORDER),
				BOX_COLOR);
		}
		cv::namedWindow("Boxes of input images");
		cv::imshow("Boxes of input images", boxed_image);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();	
}

void ImageBoxer::drawClusters(){

	//check state of local data
	checkInputImages();
	checkFeatures();
	checkClusters();
	checkBoxes();

	for (size_t i = 0; i < input_images.size(); i++){	//iterate over images
		cv::Mat boxed_image = input_images[i];
		for (size_t j = 0; j < box_coords[i].size(); j++){	//iterate over regions of interest
			cv::rectangle(boxed_image, 
				cv::Point(box_coords[i][j][TOP_X]-BOX_BORDER, box_coords[i][j][TOP_Y]-BOX_BORDER), 
				cv::Point(box_coords[i][j][BOT_X]+BOX_BORDER, box_coords[i][j][BOT_Y]+BOX_BORDER),
				BOX_COLOR);
		}
		cv::Mat image_keypoints;
		cv::drawKeypoints(boxed_image, keypoints[i], image_keypoints);
		cv::namedWindow("Clusters of keypoints of input images");
		cv::imshow("Clusters of keypoints of input images", image_keypoints);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();	
}

void ImageBoxer::showWindows(){

	//check state of local data
	checkInputImages();
	checkFeatures();
	checkClusters();
	checkBoxes();

	for (size_t i = 0; i < box_coords.size(); i++){	//iterate over images
		for (size_t j = 0; j < box_coords[i].size(); j++){	//iterate over regions of interest
			int x = box_coords[i][j][TOP_X];
			int y = box_coords[i][j][TOP_Y];
			int width = box_coords[i][j][BOT_X] - box_coords[i][j][TOP_X];
			int height = box_coords[i][j][BOT_Y] - box_coords[i][j][TOP_Y];
			int c_index = box_coords[i][j][CLUSTER_INDEX];
			cv::Mat window = input_images[i](cv::Rect(x, y, width, height));

			cv::String window_name = "Box " + std::to_string(j) + 
			" associated with cluster " + std::to_string(c_index) + 
			" of image " + std::to_string(i);
			cv::namedWindow(window_name);
			cv::imshow(window_name, window);
			cv::waitKey(0);
			cv::destroyAllWindows();
		}
	}	
}

void ImageBoxer::printFeatures(){

	//check state of local data
	checkInputImages();
	checkFeatures();
	checkClusters();
		
	for (size_t i = 0; i < clusters.size(); i++)	//iterate over images
		for (size_t j = 0; j < clusters[i].size(); j++)	//iterate over clusters
			for (size_t k = 0; k < clusters[i][j].size(); k++){	//iterate over keypoints

				cv::KeyPoint keypoint = *clusters[i][j][k].keypoint_ptr;

				//print information about the keypoint and its ORB features
				std::cout << "KeyPoint of index " + std::to_string(k) + " from cluster " +
				std::to_string(keypoint.class_id) + ", window " + std::to_string(j) +
				", image " + std::to_string(i) + " at coordinates (" + 
				std::to_string(keypoint.pt.x) + ", " + std::to_string(keypoint.pt.y) + "):" << std::endl;
				cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
				std::cout << keypoint_features << std::endl;
			}
}

std::vector< std::vector< std::vector<keypoint_t> > > ImageBoxer::getClusters(){
	return clusters;
}

//N.B. let 
//cv::KeyPoint keypoint = *clusters[i][j][k].keypoint_ptr;
//its features is given by 
//cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index); 
//a 1x32 matrix
std::vector<cv::Mat> ImageBoxer::getFeatures(){
	return features;
}

//N.B. box_coord[i][j] -> i is the image index but j is not necessarily the cluster index (e.g. box_coords[1][0] is the the 0th box of image 1, 
//and the 0th box could have all the elements from cluster 2 and not cluster 0)
std::vector< std::vector<Vec5i> > ImageBoxer::getCoords(){
	return box_coords;
}