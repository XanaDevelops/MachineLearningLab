#include "KMeans.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/PCADimensionalityReduction.h"
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map> 
using namespace System::Windows::Forms; // For MessageBox


///  KMeans class implementation  ///

// KMeans function: Constructor for KMeans class.//
KMeans::KMeans(int numClusters, int maxIterations)
	: numClusters_(numClusters), maxIterations_(maxIterations) {}


// fit function: Performs K-means clustering on the given dataset and return the centroids of the clusters.//
void KMeans::fit(const std::vector<std::vector<double>>& data) {
	// Create a copy of the data to preserve the original dataset
	std::vector<std::vector<double>> normalizedData = data;
	int die, sample_size = data.size(), num_feat = data[0].size(), iterations = 0, check;
	double mindist, dist, thresh=0.;
	bool exit=false, finish=false;
	std::vector<int> centroid_index, label, cluster(sample_size,0), cluster_size(numClusters_,0);
	std::vector<std::vector<double>> newcentroids(numClusters_,std::vector<double>(num_feat,0.));
	

	/* Implement the following:
		1 --- Initialize centroids randomly
		2 --- Randomly select unique centroid indices
		3 --- Perform K-means clustering
		4 --- Assign data points to the nearest centroid
		5 --- Calculate the Euclidean distance between the point and the current centroid
		6 --- Update newCentroids and clusterCounts
		7 --- Update centroids
		8 --- Check for convergence
	*/
	
	// TODO
	
	// 1
	std::random_device rd;  // Un dispositivo hardware per generare un seme
	std::mt19937 gen(rd()); // Mersenne Twister Engine
	std::uniform_int_distribution<> distr(0, sample_size);
	
	centroid_index.push_back(distr(gen));

	for (int i = 1; i < numClusters_; i++)
	{	
		exit = false;
		while (!exit)
		{
			die = distr(gen);
			for (int j = 0; j < i; j++)
			{
				if (die==centroid_index[j])
					break;
			}
			exit = true;
		}
		centroid_index.push_back(die);
	}

	for (int i = 0; i < numClusters_; i++)
	{
		centroids_.push_back(data[centroid_index[i]]);
	}

	// 2
	for (int i = 0; i < numClusters_; i++)
	{
		label.push_back(i);
	}


	while (!finish)
	{
		for (int k = 0; k < sample_size; k++)
		{
			mindist = SimilarityFunctions::euclideanDistance(data[k],centroids_[0]);
			cluster[k] = 0;

			for (int i = 1; i < numClusters_; i++)
			{
				dist = SimilarityFunctions::euclideanDistance(data[k], centroids_[i]);
				if (dist < mindist)
				{
					mindist = dist;
					cluster[k] = i;
				}
			}
		}

		if (iterations++ >= maxIterations_)
		{
			finish = true;
			break;
		}

		for (int i = 0; i < numClusters_; i++)
		{
			cluster_size[i] = 0;
			for (int j = 0; j < num_feat; j++)
			{
				newcentroids[i][j] = 0.;
			}
		}

		for (int k = 0; k < sample_size; k++)
		{
			//debug = cluster[k];
			for (int j = 0; j < num_feat; j++)
			{
				newcentroids[cluster[k]][j] += data[k][j];
				
			}
			cluster_size[cluster[k]]++;
		}

		for (int i = 0; i < numClusters_; i++)
		{
			for (int j = 0; j < num_feat; j++)
			{	
				
				newcentroids[i][j] = newcentroids[i][j] / cluster_size[i];
				
				
			}
		}
		
		check = 1;

		for (int i = 0; i < numClusters_; i++)
		{
			if (SimilarityFunctions::euclideanDistance(newcentroids[i],centroids_[i])>thresh)
			{
				check = 0;
				break;
			}
		}

		if (check == 1)
		{
			finish = true;
		}
		else
		{
			for (int i = 0; i < numClusters_; i++)
			{
				for (int j = 0; j < num_feat; j++)
				{
					centroids_[i][j] = newcentroids[i][j];
				}
			}
		}
		

	}

}


//// predict function: Calculates the closest centroid for each point in the given data set and returns the labels of the closest centroids.//
std::vector<int> KMeans::predict(const std::vector<std::vector<double>>& data) const {
	std::vector<int> labels;
	labels.reserve(data.size());
	double mindist, dist;
	int cluster;
	
	/* Implement the following:
		--- Initialize the closest centroid and minimum distance to the maximum possible value
		--- Iterate through each centroid
		--- Calculate the Euclidean distance between the point and the centroid
		--- Add the closest centroid to the labels vector
    */
	
	// TODO

	for (int k = 0; k < data.size(); k++)
	{
		mindist = SimilarityFunctions::euclideanDistance(centroids_[0], data[k]);
		cluster = 1;

		for (int i = 1; i < numClusters_; i++)
		{
			dist = SimilarityFunctions::euclideanDistance(centroids_[i], data[k]);

			if (dist < mindist)
			{
				mindist = dist;
				cluster = i + 1;
			}
		}

		labels.push_back(cluster);
	}

	return labels; // Return the labels vector

}





/// runKMeans: this function runs the KMeans clustering algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<int>, std::vector<std::vector<double>>>
KMeans::runKMeans(const std::string& filePath) {
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