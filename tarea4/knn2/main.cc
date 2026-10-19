#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"

#include "timer.h"

#include <cassert>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>

using namespace knn;
using namespace mnist;
using namespace crossValidation;

/// <summary>
/// Finds the best K parameter of the KNN by performing an N-Fold Cross Validation for the specified range of K.
/// </summary>
/// <param name="minK">The minimum K parameter to be tested.</param>
/// <param name="maxK">The maximum K parameter to be tested.</param>
/// <param name="dataset">The dataset to be used during the cross validation.</param>
/// <param name="folds">The number of folds.</param>
/// <returns>
/// The value of K that better performs in the cross validation.
/// </returns>
int fineTuning(unsigned int minK, unsigned int maxK, const std::vector<TSample> &dataset, unsigned int folds)
{
    assert(minK >= 1 && maxK > minK);
    
    CrossValidation crossValidation = CrossValidation::NFold(dataset, folds);
    std::cout<<"Starting search for best K parameter in range ["<<minK<<", "<<maxK<<"] using "<<crossValidation.getName()<<" with "<<dataset.size()<<" samples."<<std::endl;
    
    unsigned int bestK = minK;
    double bestAccuracy = 0.0;
    for(int k = minK; k <= maxK; k++)
    {
        std::cout<<"Testing k "<<k<<std::endl;
        double start = elapsedTime();
        
        double accuracy = crossValidation.validate(k);
        std::cout<<"K: "<<k<<" Accuracy: "<<accuracy * 100<<"% completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
        
        if(accuracy > bestAccuracy)
        {
            bestAccuracy = accuracy;
            bestK = k;
        }
    }
    
    return bestK;
}

/// <summary>
/// Loads the MNIST dataset.
/// </summary>
/// <param name="train">The output training data.</param>
/// <param name="test">The output test data.</param>
/// <returns>
/// True if the dataset was loaded successfully, false otherwise.
/// </returns>
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    // Load training data.
    std::string img_path = "train/train-images-idx3-ubyte";
    std::string label_path = "train/train-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), train))
    {
        std::cout<<"The training data could not be read"<<std::endl;
        
        return false;
    }
    
    // Load testing data.
    img_path = "test/t10k-images-idx3-ubyte";
    label_path = "test/t10k-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), test))
    {
        std::cout<<"The test data could not be read"<<std::endl;
        
        return false;
    }
    
    return true;
}

/// <summary>
/// The entry point.
/// </summary>
/// <param name="argc">The number of arguments.</param>
/// <param name="argv">The arguments.</param>
int main(int argc, char** argv) 
{
    // Load MNIST dataset.
    double totalStart = elapsedTime();
    std::vector<TSample> train;
    std::vector<TSample> test;
    if(!loadDataset(train, test))
    {
        return 1;
    }
    
    // Fine tune the KNN classifier with training data.
    double start = elapsedTime();
    int k = fineTuning(1, 10, std::vector<TSample>(train.begin(), train.begin() + std::min<std::size_t>(train.size(), 20000)), 5);
    std::cout<<"Best k: "<<k<<" found in "<<elapsedTime() - start<<" seconds"<<std::endl;
    
    // Test the classifier with testing data.
    start = elapsedTime();
    std::vector<int> labels;
    KNN knn(train, k);
    double accuracy = knn.classifyAndEvaluate(test, labels);
    
    std::cout<<"Test accuracy: "<<accuracy<<" completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
    
    std::cout<<"Total elapsed time "<<elapsedTime() - totalStart<<" seconds"<<std::endl;
    
    return 0;
}
