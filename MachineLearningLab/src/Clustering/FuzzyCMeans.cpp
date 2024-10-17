#include "FuzzyCMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  FuzzyCMeans class implementation  ///


// FuzzyCMeans function: Constructor for FuzzyCMeans class.//
FuzzyCMeans::FuzzyCMeans(int numClusters, int maxIterations, double fuzziness)
	: numClusters_(numClusters), maxIterations_(maxIterations), fuzziness_(fuzziness) {}


// fit function: Performs Fuzzy C-Means clustering on the given dataset and return the centroids of the clusters.//
void FuzzyCMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;

	/* Implement the following:
		--- Initialize centroids randomly
		--- Initialize the membership matrix with the number of data points
		--- Perform Fuzzy C-means clustering
	*/
	
	int numDataPoints = data.size(); // Get the number of data points
	int numFeatures = data[0].size(); // Get the number of features

	// Initialize centroids randomly
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	centroids_.resize(numClusters_, std::vector<double>(numFeatures, 0.0));
	for (int i = 0; i < numClusters_; ++i) {
		for (int j = 0; j < numFeatures; ++j) {
			centroids_[i][j] = dist(gen);
		}
	}

	//inicialize matrix
	initializeMembershipMatrix(numDataPoints);

	//perform actual C-means
	for (int i = 0; i < maxIterations_; ++i) {
		// Update centroids
		centroids_ = updateCentroids(data);

		// Update membership matrix
		updateMembershipMatrix(data, centroids_);
	}

	return;
}


std::vector<double> getRandValues(int size) {
	/*
		Fills a vector of size elements with random values that sum up to 1.
		Just like normalizing a vector.
	*/
	std::vector<double> values(size);
	double sum = 0;
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	for (int i = 0; i < size; i++) {
		values[i] = dist(gen);
		sum += values[i];
	}
	for (int i = 0; i < size; i++) {
		values[i] /= sum;
	}

	return values;
}

// initializeMembershipMatrix function: Initializes the membership matrix with random values that sum up to 1 for each data point.//
void FuzzyCMeans::initializeMembershipMatrix(int numDataPoints) {
	membershipMatrix_.clear();
	membershipMatrix_.resize(numDataPoints, std::vector<double>(numClusters_, 0.0));

	/* Implement the following:
		--- Initialize membership matrix with random values that sum up to 1 for each data point
		---	Normalize membership values to sum up to 1 for each data point
	*/
	//get rand values
	for (int i = 0; i < numDataPoints; i++) {
		membershipMatrix_[i] = getRandValues(numClusters_);
	}
	
}


// updateMembershipMatrix function: Updates the membership matrix using the fuzzy c-means algorithm.//
void FuzzyCMeans::updateMembershipMatrix(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>> centroids_) {

	/* Implement the following:
		---	Iterate through each data point
		--- Calculate the distance between the data point and the centroid
		--- Update the membership matrix with the new value
		--- Normalize membership values to sum up to 1 for each data point
	*/
	
	int numDataPoints = data.size(); // Get the number of data points
	int numFeatures = data[0].size(); // Get the number of features

	//iterate through each data point
	for (int i = 0; i < numDataPoints; i++) {
		//calculate the distance between the data point and the centroid
		std::vector<double> distances(numClusters_, 0.0);
		for (int j = 0; j < numClusters_; j++) {
			distances[j] = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);
		}

		//update the membership matrix with the new value
		for (int j = 0; j < numClusters_; j++) {
			double sum = 0.0;
			for (int k = 0; k < numClusters_; k++) {
				sum += std::pow(distances[j] / distances[k], 1.0 / (fuzziness_ - 1));
			}
			membershipMatrix_[i][j] = 1.0 / sum;
		}
	}
	
}


// updateCentroids function: Updates the centroids of the Fuzzy C-Means algorithm.//
std::vector<std::vector<double>> FuzzyCMeans::updateCentroids(const std::vector<std::vector<double>>& data) {

	/* Implement the following:
		--- Iterate through each cluster
		--- Iterate through each data point
		--- Calculate the membership of the data point to the cluster raised to the fuzziness
	*/

	int numDataPoints = data.size(); // Get the number of data points
	int numFeatures = data[0].size(); // Get the number of features

	//iterate clusters
	std::vector<std::vector<double>> newCentroids(numClusters_, std::vector<double>(numFeatures, 0.0));
	for (int cluster = 0; cluster < numClusters_; cluster++) {
		for (int j = 0; j < numFeatures; j++) {
			double num = 0, dem = 0;
			for (int dp = 0; dp < numDataPoints; dp++) {
				num += std::pow(membershipMatrix_[dp][cluster], fuzziness_) * data[dp][j];
				dem += std::pow(membershipMatrix_[dp][cluster], fuzziness_);

			}
			newCentroids[cluster][j] = num / dem;
		}
		
	}
	return newCentroids; // Return the centroids
}
	


	



// predict function: Predicts the cluster labels for the given data points using the Fuzzy C-Means algorithm.//
std::vector<int> FuzzyCMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels; // Create a vector to store the labels
	labels.reserve(data.size()); // Reserve space for the labels

	/* Implement the following:
		--- Iterate through each point in the data
		--- Iterate through each centroid
		--- Calculate the distance between the point and the centroid
		--- Calculate the membership of the point to the centroid
		--- Add the label of the closest centroid to the labels vector
	*/

	int numDataPoints = data.size(); // Get the number of data points
	int numFeatures = data[0].size(); // Get the number of features

	//iterate data points
	for (int i = 0; i < numDataPoints; i++) {
		//iterate centroids
		double minDistance = std::numeric_limits<double>::infinity();
		int label = -1;
		for (int j = 0; j < numClusters_; j++) {
			//calculate the distance between the point and the centroid
			double distance = SimilarityFunctions::euclideanDistance(data[i], centroids_[j]);
			//calculate the membership of the point to the centroid
			double membership = 0.0;
			for (int k = 0; k < numClusters_; k++) {
				membership += std::pow(SimilarityFunctions::euclideanDistance(data[i], centroids_[j]) / SimilarityFunctions::euclideanDistance(data[i], centroids_[k]), 1.0 / (fuzziness_ - 1));
			}
			//add the label of the closest centroid to the labels vector
			if (distance < minDistance) {
				minDistance = distance;
				label = j;
			}
		}
		labels.push_back(label+1);

	}

	return labels; // Return the labels vector

}


/// runFuzzyCMeans: this function runs the Fuzzy C-Means clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
FuzzyCMeans::runFuzzyCMeans(const std::string& filePath) {
	DataPreprocessor DataPreprocessor;
	try {
		// Check if the file path is empty
		if (filePath.empty()) {
			MessageBox::Show("Please browse and select the dataset file from your PC.");
			return {}; // Return an empty vector since there's no valid file path
		}

		// Attempt to open the file
		std::ifstream file(filePath);
		if (!file.is_open()) {
			MessageBox::Show("Failed to open the dataset file");
			return {}; // Return an empty vector since file couldn't be opened
		}

		std::vector<std::vector<double>> dataset; // Create an empty dataset vector
		DataLoader::loadAndPreprocessDataset(filePath, dataset);

		// Use the all dataset for training and testing sets.
		double trainRatio = 1.0;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData);

		// Make predictions on the training data
		std::vector<int> labels = predict(trainData);

		// Calculate evaluation metrics
		// Calculate Davies BouldinIndex using the actual features and predicted cluster labels
		double daviesBouldinIndex = Metrics::calculateDaviesBouldinIndex(trainData, labels);

		// Calculate Silhouette Score using the actual features and predicted cluster labels
		double silhouetteScore = Metrics::calculateSilhouetteScore(trainData, labels);


		// Create an instance of the PCADimensionalityReduction class
		PCADimensionalityReduction pca;

		// Perform PCA and project the data onto a lower-dimensional space
		int num_dimensions = 2; // Number of dimensions to project onto
		std::vector<std::vector<double>> reduced_data = pca.performPCA(trainData, num_dimensions);

		MessageBox::Show("Run completed");
		return std::make_tuple(daviesBouldinIndex, silhouetteScore, std::move(labels), std::move(reduced_data));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<int>(), std::vector<std::vector<double>>());
	}
}