// =============================================================================
// knn_seq.cpp  -  Version SECUENCIAL parametrizada
//
// Hace exactamente lo mismo que main.cc (mismas clases CrossValidation y KNN,
// mismo fineTuning, mismas salidas por pantalla). La unica diferencia es que el
// tamano del problema se puede elegir por linea de comandos, con los MISMOS
// argumentos que knn_mpi y knn_mpi_spec:
//
//     ./knn_seq [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]
//     por defecto: 10 5 20000 todas todas   (= el problema de main.cc)
//
// Sirve para dos cosas:
//   * Referencia secuencial AL MISMO TAMANO que las versiones MPI (speed-up).
//   * Perfilar con gprof a un tamano reducido (main.cc fija 20000 muestras).
//
// Compilar:  g++ -O2 -std=c++17 -I. knn_seq.cpp crossValidation.cc mnist.cc knn.cc sample.cc -o build/knn_seq
// =============================================================================

#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"

#include "timer.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace knn;
using namespace mnist;
using namespace crossValidation;

// Copia de fineTuning de main.cc (misma logica y mismas salidas).
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

// Igual que loadDataset de main.cc.
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    std::string img_path = "train/train-images-idx3-ubyte";
    std::string label_path = "train/train-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), train))
    {
        std::cout<<"The training data could not be read"<<std::endl;
        return false;
    }

    img_path = "test/t10k-images-idx3-ubyte";
    label_path = "test/t10k-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), test))
    {
        std::cout<<"The test data could not be read"<<std::endl;
        return false;
    }

    return true;
}

int main(int argc, char** argv)
{
    const unsigned minK = 1;
    unsigned maxK = 10, folds = 5;
    std::size_t tuningSamples = 20000;
    std::size_t testSamples = std::numeric_limits<std::size_t>::max();
    std::size_t finalTrainSamples = std::numeric_limits<std::size_t>::max();
    try
    {
        if (argc > 1) maxK = std::stoul(argv[1]);
        if (argc > 2) folds = std::stoul(argv[2]);
        if (argc > 3) tuningSamples = std::stoul(argv[3]);
        if (argc > 4) testSamples = std::stoul(argv[4]);
        if (argc > 5) finalTrainSamples = std::stoul(argv[5]);
    }
    catch (const std::exception &)
    {
        std::cerr << "Uso: knn_seq [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]" << std::endl;
        return 1;
    }
    if (maxK <= minK || folds < 2)
    {
        std::cerr << "Se necesita maxK > 1 y folds >= 2" << std::endl;
        return 1;
    }

    // Load MNIST dataset.
    double totalStart = elapsedTime();
    std::vector<TSample> train;
    std::vector<TSample> test;
    if(!loadDataset(train, test))
    {
        return 1;
    }
    const double tLoad = elapsedTime() - totalStart;

    const std::size_t nTuning = std::min(train.size(), tuningSamples);
    const std::size_t nTest = std::min(test.size(), testSamples);
    const std::size_t nFinalTrain = std::min(train.size(), finalTrainSamples);

    // Fine tune the KNN classifier with training data.
    double start = elapsedTime();
    int k = fineTuning(minK, maxK, std::vector<TSample>(train.begin(), train.begin() + nTuning), folds);
    const double tTune = elapsedTime() - start;
    std::cout<<"Best k: "<<k<<" found in "<<tTune<<" seconds"<<std::endl;

    // Test the classifier with testing data.
    start = elapsedTime();
    // Por defecto se usan los conjuntos completos (como main.cc); solo se copia un subconjunto si se pide uno menor.
    std::vector<TSample> reducedTrain, reducedTest;
    const std::vector<TSample> *finalTrain = &train;
    const std::vector<TSample> *finalTest = &test;
    if(nFinalTrain < train.size())
    {
        reducedTrain.assign(train.begin(), train.begin() + nFinalTrain);
        finalTrain = &reducedTrain;
    }
    if(nTest < test.size())
    {
        reducedTest.assign(test.begin(), test.begin() + nTest);
        finalTest = &reducedTest;
    }
    std::vector<int> labels;
    KNN knn(*finalTrain, k);
    double accuracy = knn.classifyAndEvaluate(*finalTest, labels);
    const double tFinal = elapsedTime() - start;

    std::cout<<"Test accuracy: "<<accuracy<<" completed in "<<tFinal<<" seconds"<<std::endl;

    const double total = elapsedTime() - totalStart;
    std::cout<<"Total elapsed time "<<total<<" seconds"<<std::endl;

    // Linea legible por maquina (la recoge run_all.sh). Columnas:
    // version,np,tuning,test,final_train,max_k,folds,best_k,test_acc,total_s,
    // load_s,tune_s,final_s,comm_s,nT,nE,restarts,wasted_s   (nT..wasted_s no aplican aqui)
    char csv[512];
    std::snprintf(csv, sizeof csv, "CSV,knn_seq,1,%zu,%zu,%zu,%u,%u,%d,%.6f,%.6f,%.6f,%.6f,%.6f,0.000000,,,,",
                  nTuning, nTest, nFinalTrain, maxK, folds, k, accuracy, total, tLoad, tTune, tFinal);
    std::cout<<csv<<std::endl;

    return 0;
}
