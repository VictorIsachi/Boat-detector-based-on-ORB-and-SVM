#include "feature_classifier.hpp"


FeatureClassifier::FeatureClassifier(std::vector<cv::Mat> input_images, std::vector<cv::Mat> features, 
	std::vector< std::vector< std::vector<keypoint_t> > > clusters, 
	std::vector< std::vector<Vec5i> > boxes, std::vector<int> labels){

	this->input_images = input_images;
	this->features = features;
	this->clusters = clusters;
	this->boxes = boxes;
	this->labels = labels;
}

FeatureClassifier::FeatureClassifier(std::vector<cv::Mat> input_images, std::vector<cv::Mat> features, 
	std::vector< std::vector< std::vector<keypoint_t> > > clusters, std::vector< std::vector<Vec5i> > boxes){

	this->input_images = input_images;
	this->features = features;
	this->clusters = clusters;
	this->boxes = boxes;
}

void FeatureClassifier::generateTrainingSet(){

	std::cout << "Generating training set..." << std::endl;

	int b_index, b_val;	//index and value of the binary flag (0 = no boat, 1 = boat)
	b_index = 0;

	for (size_t i = 0; i < clusters.size(); i++)	//iterate over images
		for (size_t j = 0; j < clusters[i].size(); j++){	//iterate over boxes

			b_val = labels[b_index++];	//lable of a box (1 = boat, 0 = no boat)
			int num_clu_features = clusters[i][j].size();	//number of features in a box

			//check data consistency
			int x = boxes[i][j][TOP_X];
			int y = boxes[i][j][TOP_Y];
			int width = boxes[i][j][BOT_X] - boxes[i][j][TOP_X];
			int height = boxes[i][j][BOT_Y] - boxes[i][j][TOP_Y];
			int c_index = boxes[i][j][CLUSTER_INDEX];
			if (c_index != static_cast<int>(j))
				std::cout << "\n\n\n!!!NON MATCHING INDEXES!!!(c_index = " + std::to_string(c_index) + 
				", j = " + std::to_string(static_cast<int>(j)) + ")" + "\nx = " + std::to_string(x) + 
				", y = " + std::to_string(y) + ", width = " + std::to_string(width) + ", height = " + 
				std::to_string(height) + "\n\n" << std::endl;
			
			//generate NUM_REPLICAS samples per box
			srand(SEED);
			for (int rep = 0; rep < NUM_REPLICAS; rep++){
				std::vector<cv::Mat> sample_features;	//a feature sample

				std::cout << "Generating sample " << rep << " of cluster " << j << 
				" from image " << i << std::endl;

				if (num_clu_features <= SAMPLE_SIZE){	//need to select all the features
					for (size_t k = 0; k < clusters[i][c_index].size(); k++){	//iterate over keypoints

						cv::KeyPoint keypoint = *clusters[i][c_index][k].keypoint_ptr;

						//check data consistency
						if (keypoint.class_id != c_index)
							std::cout << "\n\n\n!!!NON MATCHING CLUSTERS!!!(c_index = " + std::to_string(c_index) + 
							", class_id = " + std::to_string(keypoint.class_id) + ")" + "\nj = " + 
							std::to_string(j) + ", i = " + std::to_string(i) + ", coordinates = " + "(" + 
							std::to_string(keypoint.pt.x) + ", " + std::to_string(keypoint.pt.y) + ")" + "\n\n" << std::endl;

						//extract features associated with the keypoint and add them to the featureset
						cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
						sample_features.push_back(keypoint_features);
					}
					//pad to reach the desired sample size
					for (int pad = 0; pad < SAMPLE_SIZE - num_clu_features; pad++){
						cv::Mat padding(cv::Size(32, 1), CV_8UC1, cv::Scalar(0));
						sample_features.push_back(padding);
					}
				}
				else{	//need to select a sample of the features
					int s_size = 0;	//current size of the sample
					bool selected[clusters[i][c_index].size()] = {false};	//binary flag indicating whether a feature has been selected
					int rand_num;
					while (s_size < SAMPLE_SIZE){
						for (size_t k = 0; k < clusters[i][c_index].size(); k++){	//iterate over keypoints

							cv::KeyPoint keypoint = *clusters[i][c_index][k].keypoint_ptr;
								
							//check data consistency
							if (keypoint.class_id != c_index)
								std::cout << "\n\n\n!!!NON MATCHING CLUSTERS!!!(c_index = " + std::to_string(c_index) + 
								", class_id = " + std::to_string(keypoint.class_id) + ")" + "\nj = " + 
								std::to_string(j) + ", i = " + std::to_string(i) + ", coordinates = " + "(" + 
								std::to_string(keypoint.pt.x) + ", " + std::to_string(keypoint.pt.y) + ")" + "\n\n" << std::endl;

							//extract features associated with the keypoint and add them to the featureset
							//with probability SAMPLE_SIZE/num_clu_features
							cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
							rand_num = rand() % num_clu_features + 1;
							if ((rand_num <= SAMPLE_SIZE) && (selected[k] == false)){
								sample_features.push_back(keypoint_features);
								s_size++;
								selected[k] = true;
							}
							//terminate as soon as the SAMPLE_SIZE has been reached
							if (s_size >= SAMPLE_SIZE)
								break;
						}
					}
				}
				extended_dataset.push_back(sample_features);
				extended_labels.push_back(b_val);
			}
		}
	std::cout << "Finished generating training set...\n" << std::endl;	
}

void FeatureClassifier::exportTrainingSet(std::string filename_data, std::string filename_labels){
		
	std::cout << "Exporting training set..." << std::endl;

	std::ofstream output_data_file, output_labels_file;
	output_data_file.open(filename_data);
	output_labels_file.open(filename_labels);

	for (size_t i = 0; i < extended_dataset.size(); i++){
		//write the features
		for (size_t j = 0; j < extended_dataset[i].size()-1; j++){
				
			//concatenate the features associated with the same region of interest
			for (int k = 0; k < DIM_ORB_FEAT; k++)
				output_data_file << static_cast<int>(extended_dataset[i][j].at<unsigned char>(0, k)) << ";";
		}
		for (int k = 0; k < DIM_ORB_FEAT-1; k++)
			output_data_file << static_cast<int>(extended_dataset[i][extended_dataset[i].size()-1].at<unsigned char>(0, k)) << ";";
		output_data_file << static_cast<int>(extended_dataset[i][extended_dataset[i].size()-1].at<unsigned char>(0, 31));
		output_data_file << std::endl;
			
		//write the label
		output_labels_file << extended_labels[i] << std::endl;
	}

	output_data_file.close();
	output_labels_file.close();

	std::cout << "Finished exporting training set...\n" << std::endl;
}

void FeatureClassifier::splitNormalizeData(){

	cv::Mat feature_dataset(extended_dataset.size(), extended_dataset[0].size()*DIM_ORB_FEAT, CV_32F);	//a dataset containing all the features
	for (size_t i = 0; i < extended_dataset.size(); i++){
		for (size_t j = 0 ; j < extended_dataset[i].size(); j++){
			for (size_t k = 0; k < DIM_ORB_FEAT; k++){
				//concatenated and normalize features
				feature_dataset.at<float> (i, j*DIM_ORB_FEAT + k) = 
				extended_dataset[i][j].at<unsigned char>(0, k) / 255.0;
			}
		}
	}
	cv::Mat labels_dataset(extended_labels.size(), 1, CV_32SC1);	//a dataset containing all the labels
	for (size_t i = 0; i < extended_labels.size(); i++){
		labels_dataset.at<float>(i, 0) = extended_labels[i];
	}

	//created the normalized training dataset
	cv::Ptr<cv::ml::TrainData> tdata = cv::ml::TrainData::create(feature_dataset, cv::ml::ROW_SAMPLE, labels_dataset);
	//split it into trainig and test according to the TRAIN_TEST_RATIO
	tdata->setTrainTestSplitRatio(TRAIN_TEST_RATIO);

	svm_data = tdata;
}

void FeatureClassifier::generateSVMModel(){

	std::cout << "Training SVM model..." << std::endl;

	//generate the SVM model
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->trainAuto(svm_data);

	//compute the training and test errors
	float training_error = svm->calcError(svm_data, false, cv::noArray());
	float test_error = svm->calcError(svm_data, true, cv::noArray());

	this->svm = svm;

	std::cout << "Finished training SVM model...\n" << std::endl;
	std::cout << "Training error: " << training_error << "%" << std::endl;
	std::cout << "Test error: " << test_error << "%" << std::endl;
}

void FeatureClassifier::exportSVMModel(cv::String file_path){

	std::cout << "Saving SVM model..." << std::endl;
	svm->save(file_path);
	std::cout << "Finished saving SVM model...\n" << std::endl; 

}

void FeatureClassifier::importSVMModel(cv::String file_path){

	std::cout << "Importing SVM model..." << std::endl;
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm = cv::Algorithm::load<cv::ml::SVM>(file_path);
	this->svm = svm;
	std::cout << "Finished importing SVM model...\n" << std::endl;
}

void FeatureClassifier::generateSamples(){

	std::cout << "Generating sample set..." << std::endl;

	for (size_t i = 0; i < clusters.size(); i++)	//iterate over images
		for (size_t j = 0; j < clusters[i].size(); j++){	//iterate over boxes

			int num_clu_features = clusters[i][j].size();	//number of features in a box
				
			srand(SEED);
			std::vector<cv::Mat> sample_features;	//a feature sample
			std::cout << "Generating sample of cluster " << j << " from image " << i << std::endl;

			if (num_clu_features <= SAMPLE_SIZE){	//need to select all the features
				for (int k = 0; k < num_clu_features; k++){
					//extract features associated with the keypoint and add them to the featureset
					cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
					sample_features.push_back(keypoint_features);
				}
				//pad to reach the desired sample size
				for (int pad = 0; pad < SAMPLE_SIZE - num_clu_features; pad++){
					cv::Mat padding(cv::Size(32, 1), CV_8UC1, cv::Scalar(0));
					sample_features.push_back(padding);
				}
			}
			else{	//need to select a sample of the features
				int s_size = 0;	//current size of the sample
				bool selected[clusters[i][j].size()] = {false};	//binary flag indicating whether a feature has been selected
				int rand_num;
				while (s_size < SAMPLE_SIZE){
					for (int k = 0; k < num_clu_features; k++){
						//extract features associated with the keypoint and add them to the featureset
						//with probability SAMPLE_SIZE/num_clu_features
						cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index);
						rand_num = rand() % num_clu_features + 1;
						if ((rand_num <= SAMPLE_SIZE) && (selected[k] == false)){
							sample_features.push_back(keypoint_features);
							s_size++;
							selected[k] = true;
						}
						//terminate as soon as the SAMPLE_SIZE has been reached
						if (s_size >= SAMPLE_SIZE)
							break;
					}
				}
			}
			extended_dataset.push_back(sample_features);
		}
	//normalize the sample from [0, 255] to [0, 1]
	cv::Mat feature_dataset(extended_dataset.size(), extended_dataset[0].size()*DIM_ORB_FEAT, CV_32F);
	for (size_t i = 0; i < extended_dataset.size(); i++){
		for (size_t j = 0 ; j < extended_dataset[i].size(); j++){
			for (size_t k = 0; k < DIM_ORB_FEAT; k++){
				feature_dataset.at<float> (i, j*DIM_ORB_FEAT + k) = 
				extended_dataset[i][j].at<unsigned char>(0, k) / 255.0;
			}
		}
	}
	samples = feature_dataset;
	std::cout << "Finished generating sample set...\n" << std::endl;	
}

void FeatureClassifier::classifyImages(){
	cv::Mat predictions;	//model predictions
	svm->predict(samples, predictions);

	int box_index = 0;
	for (size_t i = 0; i < boxes.size(); i++){	//iterate over images
		cv::Mat boxed_image = input_images[i];
		for (size_t j = 0; j < boxes[i].size(); j++){	//iterate over regions of interest
				
			if (predictions.at<float>(box_index++, 0) >= 1){	//region has been classified as containg a boat
				cv::rectangle(boxed_image, 
				cv::Point(boxes[i][j][TOP_X]-BOX_BORDER, boxes[i][j][TOP_Y]-BOX_BORDER), 
				cv::Point(boxes[i][j][BOT_X]+BOX_BORDER, boxes[i][j][BOT_Y]+BOX_BORDER),
				BOX_COLOR);
			}
		}
		cv::namedWindow("Classified input images");
		cv::imshow("Classified input images", boxed_image);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();
}

void FeatureClassifier::classifyImages(cv::String file_path){
	cv::Mat predictions;	//model predictions
	svm->predict(samples, predictions);
		
	std::ifstream input_boxes;	//file containing the boat locations
	input_boxes.open(file_path);
	std::vector< std::vector<cv::Vec4i> > true_boxes;	//location of the boats ([image][boat])
	std::vector< std::vector<bool> > boxes_detected;	//keeps track of the boats detected by the system ([image][boat])

	//read the coordinates of the boats present in the image
	std::string line;
	while (std::getline(input_boxes, line)){	//iterate over lines (images)
		std::istringstream iss(line);
		//std::cout << line << std::endl;

		int x_min, x_max, y_min, y_max;
		std::vector<cv::Vec4i> line_boxes;	//boat present in current line (image)
		std::vector<bool> line_detected;	//detected boats in the line (image)

		while (iss >> x_min >> x_max >> y_min >> y_max){	//iterate over position in a line (boats)

			cv::Vec4i box(x_min, y_min, x_max, y_max);
			line_boxes.push_back(box);
			line_detected.push_back(false);
		}
		true_boxes.push_back(line_boxes);
		boxes_detected.push_back(line_detected);
	}

	/*USED TO CHECK IF THE TRUE BOXES HAVE BEEN STORED PROPERLY
	for (size_t i = 0; i < true_boxes.size(); i++){
		for (size_t j = 0; j < true_boxes[i].size(); j++){
			for (size_t k = 0; k < 4; k++){
				std::cout << true_boxes[i][j][k] << " ";
			}
			std::cout << "; ";
		}
		std::cout << std::endl;
	}*/

	double total_IoU = 0;	//total IoU score over the dataset
	int total_boxes = 0;	//total number of boats
	int box_index = 0;
	for (size_t i = 0; i < boxes.size(); i++){	//iterate over the images
		cv::Mat boxed_image = input_images[i];

		double image_IoU = 0;	//image IoU score
		int image_boxes = 0;	//number of boats in image
		for (size_t j = 0; j < boxes[i].size(); j++){	//iterate over boxes
			if (predictions.at<float>(box_index++, 0) >= 1){	//detected boat
				cv::rectangle(boxed_image, 
				cv::Point(boxes[i][j][TOP_X]-BOX_BORDER, boxes[i][j][TOP_Y]-BOX_BORDER), 
				cv::Point(boxes[i][j][BOT_X]+BOX_BORDER, boxes[i][j][BOT_Y]+BOX_BORDER),
				BOX_COLOR);

				//compute the best match among the true boat locations
				double box_IoU = 0;
				int true_box_index = -1;
				for (size_t u = 0; u < true_boxes[i].size(); u++){
					double inter, uni;

					int x_detected = boxes[i][j][TOP_X];
					int y_detected = boxes[i][j][TOP_Y];
					int width_detected = boxes[i][j][BOT_X] - boxes[i][j][TOP_X];
					int height_detected = boxes[i][j][BOT_Y] - boxes[i][j][TOP_Y];
					cv::Rect d_box(x_detected, y_detected, width_detected, height_detected);

					int x_true = true_boxes[i][u][TOP_X];
					int y_true = true_boxes[i][u][TOP_Y];
					int width_true = true_boxes[i][u][BOT_X] - true_boxes[i][u][TOP_X];
					int height_true = true_boxes[i][u][BOT_Y] - true_boxes[i][u][TOP_Y];
					cv::Rect t_box(x_true, y_true, width_true, height_true);

					cv::Rect inter_box = d_box & t_box; //intersection between boxes

					inter = inter_box.area();
					uni = d_box.area() + t_box.area() - inter;

					if (box_IoU < (inter/uni)){
						box_IoU = inter/uni;
						true_box_index = u;
					}

					/*USED TO DRAW THE BOXES
					cv::rectangle(boxed_image, t_box, cv::Scalar(200,0,0));//blue
					cv::rectangle(boxed_image, d_box, cv::Scalar(0,200,0));//green
					cv::rectangle(boxed_image, inter_box, cv::Scalar(0,0,200));//red
					std::cout<<"uni: "<<uni<<", inter: "<<inter;
					std::cout<<", IoU: "<<box_IoU<<std::endl;*/
				}
				if (true_box_index >= 0)	//there is overlap between true and detected boxes
					boxes_detected[i][true_box_index] = true;
				image_IoU += box_IoU;
				image_boxes++;
			}
		}
		total_IoU += image_IoU;
		total_boxes += image_boxes;
			
		cv::namedWindow("Classified input images");
		cv::imshow("Classified input images", boxed_image);
		cv::waitKey(0);
		std::cout << "Average image IoU: " << image_IoU/image_boxes << std::endl;
	}
	cv::destroyAllWindows();
	std::cout << "\nAverage global IoU: " << total_IoU/total_boxes << std::endl;
	input_boxes.close();
}