#include "EntropyFunctions.h"
#include <vector>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <set>
#include <unordered_set>


									// EntropyFunctions class implementation //



/// Calculates the entropy of a given set of labels "y".///
double EntropyFunctions::entropy(const std::vector<double>& y) {
	int total_samples = y.size();
	std::vector<double> labels;
	std::unordered_map<double, int> label_map;
	double entropy = 0.0;
	
	// Convert labels to unique integers and count their occurrences
	//TODO

	labels = y;
	std::sort(labels.begin(), labels.end());
	auto new_end=std::unique(labels.begin(), labels.end());
	labels.erase(new_end, labels.end());

	int total_labels = labels.size();
	std::vector<double> probability(total_labels,0);

	double add = 1.0 / total_samples;

	for (int i = 0; i < total_samples; i++)
	{
		for (int j = 0; j < total_labels; j++)
		{
			if (y[i] == labels[j])
			{
				probability[j]+=add;
				break;
			}
		}
	}


	for (int j = 0; j < total_labels; j++)
	{
		entropy  -= probability[j] * log2(probability[j]);
	}


	// Compute the probability and entropy
	//TODO

	return entropy;
}


/// Calculates the entropy of a given set of labels "y" and the indices of the labels "idxs".///
double EntropyFunctions::entropy(const std::vector<double>& y, const std::vector<int>& idxs) {
	std::vector<double> hist;
	std::unordered_map<double, int> label_map;
	int total_samples = idxs.size();
	double entropy = 0.0;
	// Convert labels to unique integers and count their occurrences
	//TODO


	// Compute the probability and entropy
	//TODO


	return entropy;
}


