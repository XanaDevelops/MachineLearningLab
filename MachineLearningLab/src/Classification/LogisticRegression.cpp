#include "LogisticRegression.h"
#include "../DataUtils/DataLoader.h"
#include "../Evaluation/Metrics.h"
#include "../DataUtils/DataPreprocessor.h"
#include <string>
#include <vector>
#include <utility>
#include <set>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <unordered_map> 

using namespace System::Windows::Forms; // For MessageBox


                                            ///  LogisticRegression class implementation  ///
// Constractor

LogisticRegression::LogisticRegression(double learning_rate, int num_epochs)
    : learning_rate(learning_rate), num_epochs(num_epochs) {}

// Fit method for training the logistic regression model
void LogisticRegression::fit(const std::vector<std::vector<double>>& X_train, const std::vector<double>& y_train) {
    
    /* Implement the following:
        1 --- Initialize weights for each class
        2 --- Loop over each class label
        3 --- Convert the problem into a binary classification problem
        4 --- Loop over training epochs
        5 --- Add bias term to the training example
        6 --- Calculate weighted sum of features
        7 --- Calculate the sigmoid of the weighted sum
        8 --- Update weights using gradient descent
    */

    int num_features = X_train[0].size();
    int num_classes = std::set<double>(y_train.begin(), y_train.end()).size();
    int num_samples = X_train.size();
    double update=0.0;
    std::vector<int> binary_y(num_samples,0);
    std::vector<double> labels = y_train, fill(num_features+1,0.5);
    std::vector<double> no_bias(num_features,0.0), scores(num_samples, 0.0), sigmoid(num_samples, 0.0);
    std::sort(labels.begin(), labels.end());
    auto new_end = std::unique(labels.begin(), labels.end());
    labels.erase(new_end, labels.end());


    // 2
    for (int i = 0; i < num_classes; i++)
    {
        weights.push_back(fill); // 1
        // 3

        for (int k = 0; k < num_samples; k++)
        {
            if (y_train[k] == labels[i])
            {
                binary_y[k] = 1;
            }
            else {
                binary_y[k] = 0;
            }
        }


        // 4
        for (int p=0; p<num_epochs; p++)

        { 
            // 5 - 6

            for (int k = 0; k < num_samples; k++)
            {   
                no_bias.erase(no_bias.begin(), no_bias.end());
                std::copy(weights[i].begin() + 1, weights[i].end(), std::back_inserter(no_bias));
                scores[k] = weights[i][0]+SimilarityFunctions::dotProduct(no_bias, X_train[k]);
            }


            // 7

            for (int k = 0; k < num_samples; k++)
            {
                sigmoid[k] = 1 / (1 + exp(-scores[k]));
            }


            // 8

            update = 0.0;

            for (int k = 0; k < num_samples; k++)
            {
                update += (sigmoid[k] - binary_y[k]); // is this right??
            }

            weights[i][0] = weights[i][0] - learning_rate / num_samples * update;


            for (int j = 1; j < num_features+1; j++)
            {
                update = 0.0;

                for (int k = 0; k < num_samples; k++)
                {
                    update += (sigmoid[k] - binary_y[k]) * X_train[k][j-1]; // is this right??
                }

                weights[i][j] = weights[i][j] - learning_rate / num_samples * update;
            }

        }
        
    }


}

// Predict method to predict class labels for test data
std::vector<double> LogisticRegression::predict(const std::vector<std::vector<double>>& X_test) {
    std::vector<double> predictions;
    
    int num_classes = weights.size();
    std::vector<std::vector<double>> scores(X_test.size(),std::vector<double>(num_classes,0.0));
    std::vector<double> no_bias;
    double max_scores;
    int max_index;

    /* Implement the following:
    	--- Loop over each test example
        --- Add bias term to the test example
        --- Calculate scores for each class by computing the weighted sum of features
        --- Predict class label with the highest score
    */
      
    // TODO

    for (int i = 0; i < X_test.size(); i++)
    {
        for (int j = 0; j < num_classes; j++)
        {   
            no_bias.erase(no_bias.begin(), no_bias.end());
            std::copy(weights[j].begin() + 1, weights[j].end(), std::back_inserter(no_bias));
            scores[i][j] = weights[j][0] + SimilarityFunctions::dotProduct(no_bias, X_test[i]);
        }

        max_scores = scores[i][0];
        max_index = 0;

        for (int j = 1; j < num_classes; j++)
        {
            if (scores[i][j] > max_scores)
            {
                max_scores = scores[i][j];
                max_index = j;
            }
        }

        predictions.push_back(max_index);
        
    }
        
    return predictions;
}

/// runLogisticRegression: this function runs the logistic regression algorithm on the given dataset and 
/// then returns a tuple containing the evaluation metrics for the training and test sets, 
/// as well as the labels and predictions for the training and test sets.///
std::tuple<double, double, std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>> 
LogisticRegression::runLogisticRegression(const std::string& filePath, int trainingRatio) {

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
        fit(trainData, trainLabels);

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