#include "LinearRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include "../Utils/SimilarityFunctions.h"
#include "../Evaluation/Metrics.h"
#include <cmath>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <unordered_map>
#include <msclr\marshal_cppstd.h>
#include <stdexcept>
#include "../MainForm.h"
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
using namespace System::Windows::Forms; // For MessageBox

#define USE_MATRIX 0

										///  LinearRegression class implementation  ///


// Function to fit the linear regression model to the training data //
void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels) {

	// This implementation is using Matrix Form method
	/* Implement the following:	  
	    --- Check if the sizes of trainData and trainLabels match
	    --- Convert trainData to matrix representation
	    --- Construct the design matrix X
		--- Convert trainLabels to matrix representation
		--- Calculate the coefficients using the least squares method
		--- Store the coefficients for future predictions
	*/
    m_coefficients = Eigen::VectorXd(1,1);
	m_coefficients.setZero();
	//checking sizes
	if (trainData.size() != trainLabels.size()) {
		MessageBox::Show("The sizes of trainData and trainLabels do not match.");
		return;
	}

	// Convert trainData to matrix representation and construct the design matrix X
	Eigen::MatrixXd X(trainData.size(), trainData[0].size() + 1);
	X.col(0) = Eigen::VectorXd::Ones(trainData.size());
	for (int i = 0; i < trainData.size(); i++) {
		for (int j = 0; j < trainData[0].size(); j++) {
			X(i, j + 1) = trainData[i][j];
		}
	}
	//convert trainLabels to matrix representation
	Eigen::VectorXd y(trainLabels.size());
	for (int i = 0; i < trainLabels.size(); i++) {
		y(i) = trainLabels[i];
	}
	// Calculate the coefficients using the least squares method
    Eigen::VectorXd coefficients = (X.transpose() * X).inverse() * X.transpose() * y; //see pdf
	m_coefficients = coefficients;


}

void LinearRegression::fit(const std::vector<std::vector<double>>& trainData, const std::vector<double>& trainLabels, double learning_rate, int num_iterations) {
    //This implementation is using Gradient Descend Method

	/* Implement the following: (copied from Logistic Regression)
        1 --- Initialize weights for each class
        2 --- Loop over each class label
        3 --- Convert the problem into a binary classification problem
        4 --- Loop over training epochs
        5 --- Add bias term to the training example
        6 --- Calculate weighted sum of features
        7 --- Calculate the sigmoid of the weighted sum
        8 --- Update weights using gradient descent
    */

	int num_samples = trainData.size();
    int num_feats = trainData[0].size();
    

    m_coefficients = Eigen::VectorXd::Zero(num_feats);

	//use gradient descent to update weights
    for (int i = 0; i < num_iterations; i++) {
        std::vector<double> gradient(num_feats, 0);
        for (int j = 0; j < num_samples; j++) {
            double prediction = 0;
            for (int k = 0; k < num_feats; k++) {
                prediction += m_coefficients[k] * trainData[j][k];
            }
            double error = trainLabels[j] - prediction;
            for (int k = 0; k < num_feats; k++) {
                gradient[k] += error * trainData[j][k];
            }

        }
        for (int j = 0; j < num_feats; j++) {
            m_coefficients[j] -= (learning_rate * gradient[j]) / num_samples;
        }
    }

}



// Function to make predictions on new data //
std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData) {

	// This implementation is using Matrix Form method    
    /* Implement the following
		--- Check if the model has been fitted
		--- Convert testData to matrix representation
		--- Construct the design matrix X
		--- Make predictions using the stored coefficients
		--- Convert predictions to a vector
	*/
	
	// TODO
    //checking size
	if (m_coefficients.size() == 1 && m_coefficients.isZero()) {
		MessageBox::Show("The model has not been fitted yet.");
        return {};
	}

	// Convert testData to matrix representation and construct the design matrix X
	Eigen::MatrixXd X(testData.size(), testData[0].size() + 1);
	X.col(0) = Eigen::VectorXd::Ones(testData.size());
	for (int i = 0; i < testData.size(); i++) {
		for (int j = 0; j < testData[0].size(); j++) {
			X(i, j + 1) = testData[i][j];
		}
	}

	// Make predictions using the stored coefficients
	Eigen::VectorXd predictions = X * m_coefficients;

	//save data to result vector
	std::vector<double> result;
	for (int i = 0; i < predictions.size(); i++) {
		result.push_back(predictions(i));
	}
	
    return result;
}

std::vector<double> LinearRegression::predict(const std::vector<std::vector<double>>& testData, int gradient) {
	//Using Gradient Descent
	/* Implement the following: (copied from Logistic Regression)
		--- Loop over each test example
		--- Add bias term to the test example
		--- Calculate scores for each class by computing the weighted sum of features
		--- Predict class label with the highest score
	*/
    std::vector<double> predictions;

    int num_features = testData[0].size();
    int num_samples = testData.size();

	for (int i = 0; i < num_samples; i++)
	{
        double prediction = 0;
		for (int j = 0; j < num_features; j++)
		{
			prediction += m_coefficients[j] * testData[i][j];
		}
        prediction += gradient;
		predictions.push_back(prediction);
	}


    return predictions;
	
}


/// runLinearRegression: this function runs the Linear Regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets. ///

std::tuple<double, double, double, double, double, double,
    std::vector<double>, std::vector<double>,
    std::vector<double>, std::vector<double>>
    LinearRegression::runLinearRegression(const std::string& filePath, int trainingRatio) {
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
#if USE_MATRIX == 1
        fit(trainData, trainLabels);
#else
		fit(trainData, trainLabels, 0.1, 1000);
#endif
        // Make predictions on the test data
#if USE_MATRIX == 1
        std::vector<double> testPredictions = predict(testData);
#else
		std::vector<double> testPredictions = predict(testData, 1);
#endif
        // Calculate evaluation metrics (e.g., MAE, MSE)
        double test_mae = Metrics::meanAbsoluteError(testLabels, testPredictions);
        double test_rmse = Metrics::rootMeanSquaredError(testLabels, testPredictions);
        double test_rsquared = Metrics::rSquared(testLabels, testPredictions);

        // Make predictions on the training data
#if USE_MATRIX == 1
        std::vector<double> trainPredictions = predict(trainData);
#else
        std::vector<double> trainPredictions = predict(trainData, 1);
#endif
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