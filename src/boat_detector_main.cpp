/*
*The software does not come with any type of checking on the correct use of its methods.
*The user is responsible for the proper use of the software and should provide data 
*as specified in the comments of the program or in the report.
*
*Quick guide: to perform a certain task (e.g. SVM model training) uncomment the MODULE
*associated with it and comment out all of the other MODULEs
*/

#include "boat_detector_utils.hpp"
#include "image_boxer.hpp"
#include "feature_classifier.hpp"
#include <string>
#include <fstream>

#define SVM_MODEL_PATH "../include/SVMModel.xml"

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
	
	//check if the path to images has been provided
	if (argc < 2){
		cout << "Error:\nThe user must at least provide the path to a folder containing images" << endl;
		return 1;
	}
	const String directory_path = argv[1];	//directory containing a set of images
	
	vector<Mat> input_images = BoatDetectorUtils::loadImages(directory_path);
	/*CHECK THE IMAGES
	BoatDetectorUtils::showImages(input_images);*/

	ImageBoxer boxed_images = ImageBoxer(input_images);
	boxed_images.extractFeatures();
	/*CHECK THE DETECTED KEYPOINTS
	boxed_images.showKeypoints();*/

	/*DOES NOT PERFORM GOOD SEGMENTATION
	boxed_images.segmentImages();
	boxed_images.showKeypointSegments();*/

	boxed_images.clusterKeypoints();
	boxed_images.boxImages();
	/*CHECK THE DETECTED CLUSTERS AND THE REGIONS OF INTEREST ASSOCIATED WITH THEM
	boxed_images.drawBoxes();
	boxed_images.showWindows();
	boxed_images.drawClusters();
	boxed_images.printFeatures();*/

	//DTA AFTER CLUSTERING AND DEFINITION OF REGIONS OF INTEREST
	vector< vector< vector<keypoint_t> > > keypoint_clusters = boxed_images.getClusters();	//clustter of keypoints([image_index][cluster_index][keypoint_index])
	vector<Mat> keypoint_features = boxed_images.getFeatures();	//features of the keypoints([global_feature_index])
	//N.B. let cv::KeyPoint keypoint = *clusters[i][j][k].keypoint_ptr;
	//its features is given by cv::Mat keypoint_features = features[i].row(clusters[i][j][k].keypoint_index); a 1x32 matrix
	vector< vector<Vec5i> > keypoint_boxes = boxed_images.getCoords();	//coordinates of the regions of interes([image][cluster]->Vec5i(top_left_x, top_left_y, bot_right_x, bot_right_y, cluster_index))

	
	/*MODULE(1) -> USED TO MANUALLY GENERATE THE BINARY LABLES FOR THE REGIONS OF INTEREST
	int window_index;	//useful to keep track of how many regions of interest have already been classified
	if (argc < 3)
		window_index = 1;
	else
		window_index = stoi(argv[2]);
	BoatDetectorUtils::test_boxer(input_images, keypoint_clusters, keypoint_features, keypoint_boxes, window_index);*/

	
	/*MODULE(2) -> USED TO MANUALLY CHECK IF THE BINARY LABESL PROVIDED IN ../include/binary_labels.txt CORRECTLY CLASSIFY THE REGIONS OF INTEREST
	ifstream infile("../include/binary_labels.txt");
	vector<int> labels;
	int temp;
	while (infile >> temp)
		labels.push_back(temp);
	BoatDetectorUtils::test_boxer(input_images, keypoint_clusters, keypoint_features, keypoint_boxes, labels);*/

	
	/*MODULE(3) -> USED TO GENERATE AN SVM MODEL GIVEN A SET OF REGIONS OF INTEREST AND THEIR BINARY LABELS
	FeatureClassifier classified_boxes = FeatureClassifier(input_images, keypoint_features, keypoint_clusters, keypoint_boxes, labels);
	classified_boxes.generateTrainingSet();
	//by default export the set of features to ../include/training.txt and the set of labels to ../include/labels.txt
	//classified_boxes.exportTrainingSet("../include/training.txt", "../include/labels.txt");	
	classified_boxes.splitNormalizeData();
	classified_boxes.generateSVMModel();
	classified_boxes.exportSVMModel(SVM_MODEL_PATH);*/

	
	/*MODULE(4) -> USED TO DETECT BOATS IN A SET OF IMAGES*/
	FeatureClassifier classified_boxes = FeatureClassifier(input_images, keypoint_features, keypoint_clusters, keypoint_boxes);
	classified_boxes.importSVMModel(SVM_MODEL_PATH);
	classified_boxes.generateSamples();
	if (argc < 3)
		classified_boxes.classifyImages();	//will not compute the IoU score
	else{
		const String true_boat_location_path = argv[2];
		classified_boxes.classifyImages(true_boat_location_path); //will compute the IoU socre
	}
	
	return 0;
}