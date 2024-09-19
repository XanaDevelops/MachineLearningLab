cmake_minimum_required(VERSION 3.29)
project(MachineLearningLab)

set(CMAKE_CXX_STANDARD 14)

include_directories(MachineLearningLab/src)
include_directories(MachineLearningLab/src/Classification)
include_directories(MachineLearningLab/src/Clustering)
include_directories(MachineLearningLab/src/DataUtils)
include_directories(MachineLearningLab/src/Evaluation)
include_directories(MachineLearningLab/src/Regression)
include_directories(MachineLearningLab/src/Utils)

find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(main PRIVATE Eigen3::Eigen)

add_executable(MachineLearningLab
    MachineLearningLab/src/Classification/DecisionTreeClassification.cpp
    MachineLearningLab/src/Classification/DecisionTreeClassification.h
    MachineLearningLab/src/Classification/KNNClassifier.cpp
    MachineLearningLab/src/Classification/KNNClassifier.h
    MachineLearningLab/src/Classification/LogisticRegression.cpp
    MachineLearningLab/src/Classification/LogisticRegression.h
    MachineLearningLab/src/Clustering/FuzzyCMeans.cpp
    MachineLearningLab/src/Clustering/FuzzyCMeans.h
    MachineLearningLab/src/Clustering/KMeans.cpp
    MachineLearningLab/src/Clustering/KMeans.h
    MachineLearningLab/src/DataUtils/DataLoader.cpp
    MachineLearningLab/src/DataUtils/DataLoader.h
    MachineLearningLab/src/DataUtils/DataPreprocessor.cpp
    MachineLearningLab/src/DataUtils/DataPreprocessor.h
    MachineLearningLab/src/Evaluation/KFoldCrossValidation.cpp
    MachineLearningLab/src/Evaluation/KFoldCrossValidation.h
    MachineLearningLab/src/Evaluation/Metrics.cpp
    MachineLearningLab/src/Evaluation/Metrics.h
    MachineLearningLab/src/Regression/DecisionTreeRegression.cpp
    MachineLearningLab/src/Regression/DecisionTreeRegression.h
    MachineLearningLab/src/Regression/KNNRegression.cpp
    MachineLearningLab/src/Regression/KNNRegression.h
    MachineLearningLab/src/Regression/LinearRegression.cpp
    MachineLearningLab/src/Regression/LinearRegression.h
    MachineLearningLab/src/Utils/EntropyFunctions.cpp
    MachineLearningLab/src/Utils/EntropyFunctions.h
    MachineLearningLab/src/Utils/Node.cpp
    MachineLearningLab/src/Utils/Node.h
    MachineLearningLab/src/Utils/PCADimensionalityReduction.cpp
    MachineLearningLab/src/Utils/PCADimensionalityReduction.h
    MachineLearningLab/src/Utils/SimilarityFunctions.cpp
    MachineLearningLab/src/Utils/SimilarityFunctions.h
    MachineLearningLab/src/MainForm.cpp
    MachineLearningLab/src/MainForm.h
    MachineLearningLab/src/attempts.cpp)

