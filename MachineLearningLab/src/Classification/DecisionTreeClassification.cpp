#include "DecisionTreeClassification.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../Utils/EntropyFunctions.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include "../DataUtils/DataPreprocessor.h"
using namespace System::Windows::Forms; // For MessageBox

// DecisionTreeClassification class implementation //


// DecisionTreeClassification is a constructor for DecisionTree class.//
DecisionTreeClassification::DecisionTreeClassification(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr) {}


// Fit is a function to fits a decision tree to the given data.//
void DecisionTreeClassification::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y); //should't here be (X, y, 0)  ???
}


// Predict is a function that Traverses the decision tree and returns the prediction for a given input vector.//
std::vector<double> DecisionTreeClassification::predict(std::vector<std::vector<double>>& X) {
	std::vector<double> predictions;
	
	// Implement the function
	// TODO
	for (int i = 0; i < X.size(); i++)
	{
		predictions.push_back(DecisionTreeClassification::traverseTree(X[i], root));
	}
	
	return predictions;
}


// growTree function: This function grows a decision tree using the given data and labelsand  return a pointer to the root node of the decision tree.//
Node* DecisionTreeClassification::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {
	
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- greedily select the best split according to information gain
		--- grow the children that result from the split
	*/
	
	double best_gain = -1.0, gain; // set the best gain to -1
	int split_idx = NULL; // split index
	double split_thresh = NULL; // split threshold
	int num_feat;

	double value;
	std::vector<double> labels=y, y_left,y_right;
	std::vector<std::vector<double>> X_left, X_right;
	// TODO

	Node* left; // grow the left tree
	Node* right;  // grow the right tree

	// stopping criteria
	std::sort(labels.begin(), labels.end());
	std::unique(labels.begin(),labels.end());
	num_feat = X[0].size();

	if ((depth >= max_depth) || (X.size() < min_samples_split) || (labels.size() == 1) || (num_feat == 1))
	{
		left = nullptr;
		right = nullptr;
		
		value = DecisionTreeClassification::mostCommonlLabel(y);

		return new Node(split_idx, split_thresh, left, right, value);
	}
	
	std::vector<double> column(X.size(), 0), values;

	for (int j = 0; j < num_feat; j++) // cycle on the features
	{
		for (int k = 0; k < X.size(); k++)
		{
			column[k] = X[k][j];
		}
		values = column;
		std::sort(values.begin(), values.end());
		std::unique(values.begin(), values.end());

		for (int k = 0; k < values.size()-1; k++)
		{
			gain = DecisionTreeClassification::informationGain(y, column, values[k]);
			if (gain > best_gain)
			{
				best_gain = gain;
				split_idx = j;
				split_thresh = values[k];
			}
		}
	}

	std::vector<double> newrow(num_feat - 1, 0);

	for (int i = 0; i < X.size(); i++)
	{

		for (int j = 0; j < split_idx; j++) // eliminate feature from X
		{
			newrow[j] = X[i][j];
		}
		for (int j = split_idx; j < num_feat - 1; j++)
		{
			newrow[j] = X[i][j - 1];
		}

		if (X[i][split_idx] <= split_thresh)
		{
			y_left.push_back(y[i]);
			X_left.push_back(newrow);
		}
		else
		{
			y_right.push_back(y[i]);
			X_right.push_back(newrow);
		}
	}

	
	left = growTree(X_left, y_left, depth + 1);
	right = growTree(X_right, y_right, depth + 1);


	return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// informationGain function: Calculates the information gain of a given split threshold for a given feature column.
double DecisionTreeClassification::informationGain(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) {
	// parent loss // You need to caculate entropy using the EntropyFunctions class//
	double parent_entropy = EntropyFunctions::entropy(y);
	double entropy_a, entropy_b, child_entropy;
	double length=y.size();

	/* Implement the following:
	   --- generate split
	   --- compute the weighted avg. of the loss for the children
	   --- information gain is difference in loss before vs. after split
	*/
	double ig = 0.0;
	
	std::vector<double> a, b;
	for (int i = 0; i < y.size(); i++)
	{
		if (X_column[i] <= split_thresh)
		{
			a.push_back(y[i]);
		}
		else {
			b.push_back(y[i]);
		}
	}

	entropy_a = EntropyFunctions::entropy(a);
	entropy_b = EntropyFunctions::entropy(b);
	child_entropy = (a.size()*entropy_a+b.size()*entropy_b)/length;

	ig = parent_entropy - child_entropy;
	// TODO
	
	return ig;
}


// mostCommonlLabel function: Finds the most common label in a vector of labels.//
double DecisionTreeClassification::mostCommonlLabel(std::vector<double>& y) {	
	int most_common_quantity = 0, most_common_index = 0;
	
	// TODO
	std::vector<double> labels;
	labels = y;
	std::sort(labels.begin(), labels.end());
	std::unique(labels.begin(), labels.end());
	std::vector<int> quantity(labels.size(), 0);

	for (int i = 0; i < y.size(); i++)
	{
		for (int j = 0; j < labels.size(); j++)
		{
			if (y[i] == labels[j])
			{
				quantity[j]++;
				break;
			}
		}
	}

	for (int k = 0; k < labels.size(); k++)
	{
		if (quantity[k] > most_common_quantity)
		{
			most_common_quantity = quantity[k];
			most_common_index = k;
		}
	}
	

	return labels[most_common_index];
}


// traverseTree function: Traverses a decision tree given an input vector and a node.//
double DecisionTreeClassification::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
	// TODO
	
	if (node->isLeafNode())
	{
		return node->value;
	}
	else if (x[node->feature]<=node->threshold)
	{
		return DecisionTreeClassification::traverseTree(x, node->left);
	}
	else 
	{
		return DecisionTreeClassification::traverseTree(x, node->right);
	}


	return 0.0;
}


/// runDecisionTreeClassification: this function runs the decision tree classification algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>>
DecisionTreeClassification::runDecisionTreeClassification(const std::string& filePath, int trainingRatio) {
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

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);//

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate accuracy using the true labels and predicted labels for the test data
		double test_accuracy = Metrics::accuracy(testLabels, testPredictions);


		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate accuracy using the true labels and predicted labels for the training data
		double train_accuracy = Metrics::accuracy(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(train_accuracy, test_accuracy,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, std::vector<double>(),
			std::vector<double>(), std::vector<double>(),
			std::vector<double>());
	}
}