// =============================================================================
// knn_mpi.cpp  -  PASO 1: paralelismo de DATOS con MPI
//
// Versión paralela de main.cc (ajuste de K por validación cruzada + evaluación
// final con el mejor K). El código secuencial NO se modifica: esta versión
// reutiliza KNN, loadMnist y TSample tal cual.
//
// Estrategia
//   * Modelo: paso de mensajes, memoria distribuida. Cada proceso MPI tiene su
//     propia memoria, así que cada uno lee el dataset de disco (no hay que
//     enviarlo por la red).
//   * Ajuste de K: hay numK x folds = 10 x 5 = 50 tareas (k, fold)
//     independientes. Se reparten de forma CÍCLICA entre los procesos
//     (tarea t -> proceso t % size). Todas cuestan casi lo mismo.
//   * Los 50 resultados se juntan con UN SOLO MPI_Allreduce. Así todos los
//     procesos conocen todas las accuracies y eligen el mismo K sin más mensajes.
//   * Evaluación final: las muestras de test se reparten en bloques contiguos;
//     cada proceso cuenta sus aciertos y MPI_Reduce los suma en el proceso 0.
//
// Compilar:   mpic++ -O2 -std=c++17 -I. knn_mpi.cpp knn.cc mnist.cc sample.cc -o build/knn_mpi
// Ejecutar (desde la carpeta build/, donde están train/ y test/):
//             mpirun -np 4 ./knn_mpi
// Parámetros opcionales (para probar otros tamaños del problema):
//             mpirun -np 4 ./knn_mpi [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]
//             por defecto: 10 5 20000 todas todas (= problema completo)
//             Con el problema completo el ajuste pesa ~84 % y la evaluacion final ~16 %.
//             En pruebas pequenas, reduce tambien [entrenamiento_final] (p. ej. 4000 1000 12000)
//             para conservar una proporcion parecida.
// =============================================================================

#include <mpi.h>

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace knn;
using namespace mnist;

namespace
{

// Aborta TODOS los procesos. Si solo terminara uno, los demás se quedarían
// bloqueados para siempre en la siguiente operación colectiva.
[[noreturn]] void abortAll(int rank, const std::string &msg)
{
    std::cerr << "[proceso " << rank << "] " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

// Igual que loadDataset de main.cc: las rutas son relativas al directorio
// desde el que se lanza el programa (en todos los nodos).
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    return loadMnist("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", train)
        && loadMnist("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", test);
}

// Número de muestras de samples[begin, end) que el clasificador acierta.
long long countMatches(const KNN &knn, const std::vector<TSample> &samples,
                       std::size_t begin, std::size_t end)
{
    long long matches = 0;
    for (std::size_t i = begin; i < end; i++)
    {
        if (knn.classify(samples[i]) == samples[i].label)
            matches++;
    }
    return matches;
}

// Accuracy del KNN con 'k' vecinos usando el fold 'fold' como validación y el
// resto como entrenamiento. Misma partición que CrossValidation::splitDataset
// del código secuencial (mismos índices, mismo orden), así que los resultados
// son idénticos a los del secuencial.
double evaluateFold(const std::vector<TSample> &data, unsigned folds, unsigned fold, int k)
{
    const std::size_t start = data.size() * fold / folds;
    const std::size_t end = data.size() * (fold + 1) / folds;

    std::vector<TSample> train;
    train.reserve(data.size() - (end - start));
    train.insert(train.end(), data.begin(), data.begin() + start);
    train.insert(train.end(), data.begin() + end, data.end());

    KNN knn(train, k);
    return countMatches(knn, data, start, end) / (double)(end - start);
}

} // namespace

int main(int argc, char **argv)
{
    // ---- Arranque de MPI --------------------------------------------------
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // identificador de este proceso: 0 .. size-1
    // Para gprof (binarios compilados con -pg): cada proceso escribe su propio fichero gmon_rank<N>.<pid>
    // en vez de pisarse todos en gmon.out. En los binarios normales esta variable no se usa.
    { char gp[32]; std::snprintf(gp, sizeof gp, "gmon_rank%d", rank); setenv("GMON_OUT_PREFIX", gp, 1); }
    MPI_Comm_size(MPI_COMM_WORLD, &size); // número total de procesos

    // ---- Parámetros del problema (por defecto, los de main.cc) -------------
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
        abortAll(rank, "Uso: knn_mpi [maxK] [folds] [muestras_ajuste] [muestras_test] [entrenamiento_final]");
    }
    if (maxK <= minK || folds < 2)
        abortAll(rank, "Se necesita maxK > 1 y folds >= 2");

    // Todos los procesos empiezan a medir a la vez.
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // ---- 1. Carga del dataset (la hace cada proceso, sin comunicación) ------
    std::vector<TSample> train, test;
    if (!loadDataset(train, test))
        abortAll(rank, "No se pudo leer MNIST: faltan las carpetas train/ y test/ en el directorio actual");
    const double tLoad = MPI_Wtime() - tStart;

    // Subconjunto de entrenamiento usado para el ajuste (igual que main.cc).
    const std::size_t nTuning = std::min(train.size(), tuningSamples);
    if (folds > nTuning)
        abortAll(rank, "Hay menos muestras de ajuste que folds");
    const std::vector<TSample> tuning(train.begin(), train.begin() + nTuning);

    // ---- 2. Ajuste de K: 50 tareas (k, fold) repartidas cíclicamente --------
    const int numK = maxK - minK + 1;
    const int numTasks = numK * folds;
    std::vector<double> acc(numTasks, 0.0); // acc[(k - minK) * folds + fold]
    int myTasks = 0;

    double t = MPI_Wtime();
    for (int task = rank; task < numTasks; task += size)
    {
        const int k = minK + task / folds;
        const int fold = task % folds;
        acc[task] = evaluateFold(tuning, folds, fold, k);
        myTasks++;
    }
    const double tTuneCompute = MPI_Wtime() - t;

    // Cada proceso solo ha rellenado SUS posiciones (el resto vale 0.0).
    // Sumando elemento a elemento obtenemos el vector completo y exacto
    // (x + 0.0 + 0.0 ... == x). Allreduce = Reduce + difusión del resultado.
    t = MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE, acc.data(), numTasks, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double tTuneComm = MPI_Wtime() - t; // incluye esperar al proceso más lento

    // Todos los procesos eligen el K de la misma forma que el secuencial:
    // media de los folds en orden, y gana el mayor accuracy (en empate, el K menor).
    std::vector<double> accPerK(numK, 0.0);
    unsigned bestK = minK;
    double bestAccuracy = 0.0;
    for (int ki = 0; ki < numK; ki++)
    {
        double a = 0.0;
        for (unsigned f = 0; f < folds; f++)
            a += acc[ki * folds + f];
        a /= folds;
        accPerK[ki] = a;
        if (a > bestAccuracy)
        {
            bestAccuracy = a;
            bestK = minK + ki;
        }
    }

    // ---- 3. Evaluación final con el mejor K: bloques de muestras de test ----
    t = MPI_Wtime();
    // Entrenamiento de la evaluacion final: por defecto, todo el conjunto.
    const std::size_t nFinalTrain = std::min(train.size(), finalTrainSamples);
    std::vector<TSample> reducedTrain;
    const std::vector<TSample> *finalTrain = &train;
    if (nFinalTrain < train.size())
    {
        reducedTrain.assign(train.begin(), train.begin() + nFinalTrain);
        finalTrain = &reducedTrain;
    }
    const KNN knn(*finalTrain, bestK);
    const std::size_t nTest = std::min(test.size(), testSamples);
    const std::size_t begin = nTest * static_cast<std::size_t>(rank) / size;
    const std::size_t end = nTest * static_cast<std::size_t>(rank + 1) / size;
    const long long matches = countMatches(knn, test, begin, end);
    const double tFinalCompute = MPI_Wtime() - t;

    t = MPI_Wtime();
    long long totalMatches = 0;
    MPI_Reduce(&matches, &totalMatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    const double tFinalComm = MPI_Wtime() - t;

    // Fin: el tiempo total es el del proceso que termina el último.
    MPI_Barrier(MPI_COMM_WORLD);
    const double tTotal = MPI_Wtime() - tStart;

    // ---- 4. Recogida de tiempos en el proceso 0 ----------------------------
    // Para cada fase nos interesa el MÁXIMO entre procesos.
    double mine[6] = {tLoad, tTuneCompute, tTuneComm, tFinalCompute, tFinalComm, tTotal};
    double worst[6];
    MPI_Reduce(mine, worst, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Tiempo de cómputo del ajuste de cada proceso (para ver el equilibrio de carga).
    std::vector<double> tuneTimes(size);
    std::vector<int> tuneTasks(size);
    MPI_Gather(&tTuneCompute, 1, MPI_DOUBLE, tuneTimes.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&myTasks, 1, MPI_INT, tuneTasks.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::cout << "Procesos MPI: " << size << " | K en [" << minK << ", " << maxK << "], " << folds
                  << " folds, " << nTuning << " muestras de ajuste, " << nTest << " de test, "
                  << nFinalTrain << " de entrenamiento final" << std::endl;
        // Mismo formato que el secuencial (6 cifras significativas) para poder compararlos.
        for (int ki = 0; ki < numK; ki++)
            std::cout << "K: " << minK + ki << " Accuracy: " << accPerK[ki] * 100 << "%" << std::endl;
        std::cout << "Best k: " << bestK << std::endl;
        std::cout << "Test accuracy: " << totalMatches / (double)nTest << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Tiempos (s, maximo entre procesos):" << std::endl;
        std::cout << "  carga del dataset (replicada)   : " << worst[0] << std::endl;
        std::cout << "  ajuste de K - calculo           : " << worst[1] << std::endl;
        std::cout << "  ajuste de K - espera + Allreduce: " << worst[2] << std::endl;
        std::cout << "  evaluacion final - calculo      : " << worst[3] << std::endl;
        std::cout << "  evaluacion final - Reduce       : " << worst[4] << std::endl;
        std::cout << "Total elapsed time " << worst[5] << " seconds" << std::endl;

        std::cout << "Reparto del ajuste por proceso (tareas / segundos de calculo):" << std::endl;
        for (int r = 0; r < size; r++)
            std::cout << "  proceso " << r << ": " << tuneTasks[r] << " tareas, " << tuneTimes[r] << " s" << std::endl;

        // Linea legible por maquina (la recoge run_all.sh). Columnas:
        // version,np,tuning,test,final_train,max_k,folds,best_k,test_acc,total_s,
        // load_s,tune_s,final_s,comm_s,nT,nE,restarts,wasted_s   (nT..wasted_s no aplican aqui)
        char csv[512];
        std::snprintf(csv, sizeof csv, "CSV,knn_mpi,%d,%zu,%zu,%zu,%u,%u,%u,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,,,,",
                      size, nTuning, nTest, nFinalTrain, maxK, folds, bestK, totalMatches / (double)nTest,
                      worst[5], worst[0], worst[1], worst[3], worst[2] + worst[4]);
        std::cout << csv << std::endl;
    }

    MPI_Finalize();
    return 0;
}
