#include "DecisionTreeRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <numeric>
#include <unordered_set>
#include "../Utils/EntropyFunctions.h"
using namespace System::Windows::Forms; // For MessageBox



///  DecisionTreeRegression class implementation  ///


// Constructor for DecisionTreeRegression class.//
DecisionTreeRegression::DecisionTreeRegression(int min_samples_split, int max_depth, int n_feats)
	: min_samples_split(min_samples_split), max_depth(max_depth), n_feats(n_feats), root(nullptr)
{
}


// fit function:Fits a decision tree regression model to the given data.//
void DecisionTreeRegression::fit(std::vector<std::vector<double>>& X, std::vector<double>& y) {
	n_feats = (n_feats == 0) ? X[0].size() : min(n_feats, static_cast<int>(X[0].size()));
	root = growTree(X, y);
}


// predict function:Traverses the decision tree and returns the predicted value for a given input vector.//
std::vector<double> DecisionTreeRegression::predict(std::vector<std::vector<double>>& X) {

	std::vector<double> predictions;
	
	// Implement the function
	// TODO
	for (int i = 0; i < X.size(); i++) {
		predictions.push_back(traverseTree(X[i], root));
	}
	return predictions;
}


// growTree function: Grows a decision tree regression model using the given data and parameters //
Node* DecisionTreeRegression::growTree(std::vector<std::vector<double>>& X, std::vector<double>& y, int depth) {

	int split_idx = -1;
	double split_thresh = 0.0;

	Node* left = nullptr;
	Node* right = nullptr;
	/* Implement the following:
		--- define stopping criteria
    	--- Loop through candidate features and potential split thresholds.
		--- Find the best split threshold for the current feature.
		--- grow the children that result from the split
	*/
	
	//X -> array of points
	//y -> labels for each d

	//define stopping criteria
	if (depth > max_depth || y.size() < min_samples_split) {
		return new Node(0,0, nullptr, nullptr, mean(y));
	}

	int best_idx = -1;
	double best_thrs = -1;
	std::vector<std::vector<double>> best_X_left, best_X_right;
	std::vector<double> best_y_left, best_y_right;

	double maxSE = std::numeric_limits<double>::infinity();
	for (int x_idx = 0; x_idx < n_feats; x_idx++) { //default is 0 so fit() sets to X[0].size()
		
		
		std::vector<double> X_Column;
		//get column values
		for (std::vector<double> X_val : X) {
			X_Column.push_back(X_val[x_idx]);
		}


		for (double split_val : X_Column) { //threshold
			std::vector<std::vector<double>> left_val, right_val;
			std::vector<double> left_y, right_y;
			//split
			for (int i = 0; i < X.size();i++) {
				if (X[i][x_idx] <= split_val) {
					left_val.push_back(X[i]);
					left_y.push_back(y[i]);
				}
				else {
					right_val.push_back(X[i]);
					right_y.push_back(y[i]);
				}
			}
			//check not empty

			if (left_val.size() == 0 || right_val.size() == 0) {
				continue;
			}
			//calculate mse
			

			double mse = meanSquaredError(y, X_Column, split_val);
			if (mse < maxSE) {
				best_idx = x_idx;
				best_thrs = split_val;
				best_X_left = left_val;
				best_X_right = right_val;
				best_y_left = left_y;
				best_y_right = right_y;
				maxSE = mse;
			}

		}	
	}

	if (maxSE > 0 && maxSE != std::numeric_limits<double>::infinity()) {
		left = growTree(best_X_left, best_y_left, depth + 1);
		right = growTree(best_X_right, best_y_right, depth + 1);

		return new Node(best_idx, best_thrs, left, right);
	}
	else {
		return new Node(0, 0, nullptr, nullptr, mean(y));
	}



	
	//return new Node(split_idx, split_thresh, left, right); // return a new node with the split index, split threshold, left tree, and right tree
}


/// meanSquaredError function: Calculates the mean squared error for a given split threshold.
double DecisionTreeRegression::meanSquaredError(std::vector<double>& y, std::vector<double>& X_column, double split_thresh) { //investigate split_tresh

	// Initialize variables to hold the sum and count for left and right regions
	double sum_left = 0, sum_right = 0;
	int count_left = 0, count_right = 0;

	// Partition data into two groups based on the split threshold
	for (size_t i = 0; i < X_column.size(); ++i) {
		if (X_column[i] <= split_thresh) {
			sum_left += y[i];
			count_left++;
		}
		else {
			sum_right += y[i];
			count_right++;
		}
	}

	// Calculate means for left and right groups
	double mean_left = (count_left > 0) ? sum_left / count_left : 0;
	double mean_right = (count_right > 0) ? sum_right / count_right : 0;

	// Compute MSE for both groups
	double mse = 0;
	for (size_t i = 0; i < X_column.size(); ++i) {
		if (X_column[i] <= split_thresh) {
			mse += (y[i] - mean_left) * (y[i] - mean_left);
		}
		else {
			mse += (y[i] - mean_right) * (y[i] - mean_right);
		}
	}

	// Return the total MSE, normalized by the number of elements
	return mse / y.size();
}

// mean function: Calculates the mean of a given vector of doubles.//
double DecisionTreeRegression::mean(std::vector<double>& values) {

	double meanValue = 0.0;
	
	// calculate the mean
	for (int i = 0; i < values.size(); i++) {
		meanValue += values[i];
	}
	meanValue /= values.size();

	return meanValue;
}

// traverseTree function: Traverses the decision tree and returns the predicted value for the given input vector.//
double DecisionTreeRegression::traverseTree(std::vector<double>& x, Node* node) {

	/* Implement the following:
		--- If the node is a leaf node, return its value
		--- If the feature value of the input vector is less than or equal to the node's threshold, traverse the left subtree
		--- Otherwise, traverse the right subtree
	*/
	// TODO
	if (node->isLeafNode()) {
		return node->value;
	}

	if (x[node->feature] <= node->threshold) {
		return traverseTree(x, node->left);
	}
	else {
		return traverseTree(x, node->right);
	}
}


/// runDecisionTreeRegression: this function runs the Decision Tree Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.

std::tuple<double, double, double, double, double, double,
	std::vector<double>, std::vector<double>,
	std::vector<double>, std::vector<double>>
	DecisionTreeRegression::runDecisionTreeRegression(const std::string& filePath, int trainingRatio) {
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
		// Load the dataset from the file path
		std::vector<std::vector<std::string>> data = DataLoader::readDatasetFromFilePath(filePath);

		// Convert the dataset from strings to doubles
		std::vector<std::vector<double>> dataset;
		bool isFirstRow = true; // Flag to identify the first row

		for (const auto& row : data) {
			if (isFirstRow) {
				isFirstRow = false;
				continue; // Skip the first row (header)
			}

			std::vector<double> convertedRow;
			for (const auto& cell : row) {
				try {
					double value = std::stod(cell);
					convertedRow.push_back(value);
				}
				catch (const std::exception& e) {
					// Handle the exception or set a default value
					std::cerr << "Error converting value: " << cell << std::endl;
					// You can choose to set a default value or handle the error as needed
				}
			}
			dataset.push_back(convertedRow);
		}

		// Split the dataset into training and test sets (e.g., 80% for training, 20% for testing)
		double trainRatio = trainingRatio * 0.01;

		std::vector<std::vector<double>> trainData;
		std::vector<double> trainLabels;
		std::vector<std::vector<double>> testData;
		std::vector<double> testLabels;

		DataPreprocessor::splitDataset(dataset, trainRatio, trainData, trainLabels, testData, testLabels);

		// Fit the model to the training data
		fit(trainData, trainLabels);

		// Make predictions on the test data
		std::vector<double> testPredictions = predict(testData);

		// Calculate evaluation metrics (e.g., MAE, MSE)
		double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
		double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
		double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

		// Make predictions on the training data
		std::vector<double> trainPredictions = predict(trainData);

		// Calculate evaluation metrics for training data
		double train_mae = Metrics::meanAbsoluteError(trainLabels, trainPredictions);
		double train_rmse = Metrics::rootMeanSquaredError(trainLabels, trainPredictions);
		double train_rsquared = Metrics::rSquared(trainLabels, trainPredictions);

		MessageBox::Show("Run completed");
		return std::make_tuple(test_mae, test_rmse, test_rsquared,
			train_mae, train_rmse, train_rsquared,
			std::move(trainLabels), std::move(trainPredictions),
			std::move(testLabels), std::move(testPredictions));
	}
	catch (const std::exception& e) {
		// Handle the exception
		MessageBox::Show("Not Working");
		std::cerr << "Exception occurred: " << e.what() << std::endl;
		return std::make_tuple(0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
			std::vector<double>(), std::vector<double>(),
			std::vector<double>(), std::vector<double>());
	}
}

