#include "SimilarityFunctions.h"
#include <cmath>
#include <stdexcept>


										// SimilarityFunctions class implementation //
			

/// hammingDistance function: Calculates the Hamming distance between two vectors.
double SimilarityFunctions::hammingDistance(const std::vector<double>& v1, const std::vector<double>& v2) {
	if (v1.size() != v2.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Hamming Distance
	for (int i = 0; i < v1.size(); i++)
	{
		if (v1[i] != v2[i]) //diferent values
		{
			dist++;
		}
	}

	return dist;
}


/// jaccardDistance function: Calculates the Jaccard distance between two vectors.
double SimilarityFunctions::jaccardDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double num = 0.0;
	double den = 0.0;
	double dist = 0.0;
	
	// Compute the Jaccard Distance
	// TODO

	return dist;
}


/// cosineDistance function: Calculates the cosine distance between two vectors.///
double SimilarityFunctions::cosineDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dotProduct = 0.0;
	double normA = 0.0;
	double normB = 0.0;
	double cosinedist = 0.0;

	std::vector<double> zero(a.size(), 0);
	normA = SimilarityFunctions::euclideanDistance(a, zero);
	normB = SimilarityFunctions::euclideanDistance(b, zero);

	for (int i = 0; i < a.size(); i++)
	{
		dotProduct += a[i] * b[i];
	}

	cosinedist = dotProduct / (normA * normB);
	
	// Compute the cosine Distance

	
	return cosinedist;
}


/// euclideanDistance function: Calculates the Euclidean distance between two vectors.///
double SimilarityFunctions::euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;

	for (int i = 0; i < a.size(); i++)
	{
		dist += pow(a[i] - b[i], 2);
	}

	dist = sqrt(dist);

	
	// Compute the Euclidean Distance
	// TODO
	

	return dist;
}


/// manhattanDistance function: Calculates the Manhattan distance between two vectors.///
double SimilarityFunctions::manhattanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Manhattan Distance
	for (int i = 0; i < a.size(); i++)
	{
		dist += abs(a[i] - b[i]);
	}

	return dist;
}

/// minkowskiDistance function: Calculates the Minkowski distance between two vectors.///
double SimilarityFunctions::minkowskiDistance(const std::vector<double>& a, const std::vector<double>& b, int p) {
	if (a.size() != b.size()) {
		throw std::invalid_argument("Vectors must be of equal length.");
	}
	double dist = 0.0;
	
	// Compute the Minkowski Distance
	for (int i = 0; i < a.size(); i++)
	{
		dist += pow(abs(a[i] - b[i]), p);
	}
	
	dist = pow(dist, 1.0 / p);

	return dist;
}


