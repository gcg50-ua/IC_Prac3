// =============================================================================
// knn_mpi_spec.cpp  -  PASOS 2 y 3: paralelismo FUNCIONAL + ejecucion ESPECULATIVA
//
// Secuencial:  ajuste de K (validacion cruzada)  ->  elegir K  ->  evaluacion final.
// La evaluacion final depende del K que salga del ajuste. Aqui se rompe esa
// dependencia "apostando": la evaluacion final EMPIEZA ya, con el mejor K
// conocido hasta ese momento, mientras el ajuste sigue ejecutandose.
//
// Dos grupos de procesos (MPI_Comm_split) que trabajan A LA VEZ:
//
//   GRUPO DE AJUSTE (rangos 0 .. nT-1)
//     Calcula las tareas (k, fold) y envia el resultado de cada una, segun la
//     acaba, al LIDER del grupo de evaluacion (MPI_Isend, tag TAG_RESULT).
//
//   GRUPO DE EVALUACION (rangos nT .. size-1; el lider es el rango nT)
//     Cada proceso clasifica su bloque de muestras de test con el mejor K
//     conocido (al principio, un K inicial supuesto). Cada pocas muestras mira
//     si ha llegado algun mensaje (MPI_Iprobe, no bloqueante):
//       - El lider recibe resultados del ajuste, detecta cuando se completa un K
//         (sus 5 folds) y, si es mejor que todos los anteriores, avisa al resto
//         del grupo con un mensaje (K, definitivo?) (tag TAG_BEST).
//       - Quien se entera de que su K ya no es el mejor DESCARTA lo hecho y
//         REINICIA con el K nuevo.
//     Al llegar el ultimo resultado, el lider envia el K definitivo. La
//     evaluacion termina cuando el bloque se ha clasificado con ese K.
//     Al final MPI_Reduce (en el comunicador del grupo) suma los aciertos.
//
// MODO (ultimo argumento)
//   0 = ESTATICO (paso 2): las tareas de ajuste se reparten de forma ciclica al
//       empezar y los procesos de evaluacion solo evaluan (y esperan).
//   1 = DINAMICO (paso 3): el lider tiene una cola de tareas de ajuste y las
//       reparte BAJO DEMANDA (TAG_REQ -> TAG_TASK). Asi el ajuste se equilibra
//       solo (no importa que P no divida a 50) y, ademas, los procesos de
//       evaluacion que terminan su bloque y esperan AYUDAN con tareas de ajuste
//       en vez de quedarse parados. Una tarea de ayuda se termina antes de
//       atender un cambio de K (para no desperdiciarla).
//
// La eleccion de K es DETERMINISTA (mayor accuracy; en empate, el K menor), asi
// que el resultado final es identico al secuencial aunque los resultados lleguen
// desordenados.
//
// Cuantos procesos van a cada grupo: por defecto, proporcional al trabajo de
// cada fase (con el problema completo, el ajuste ~84 % y la evaluacion ~16 %).
//
// Compilar:   mpic++ -O2 -std=c++17 -I. knn_mpi_spec.cpp knn.cc mnist.cc sample.cc -o build/knn_mpi_spec
// Ejecutar (desde build/):  mpirun -np 4 ./knn_mpi_spec     (minimo 2 procesos)
// Parametros opcionales:
//   mpirun -np 4 ./knn_mpi_spec [maxK] [folds] [muestras_ajuste] [muestras_test]
//                               [entrenamiento_final] [procesos_evaluacion] [k_inicial] [modo]
//   por defecto: 10 5 20000 todas todas 0(=automatico) 1 0
// =============================================================================

#include <mpi.h>

#include "sample.h"
#include "knn.h"
#include "mnist.h"

#include <unistd.h> // usleep

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace knn;
using namespace mnist;

namespace
{

const int TAG_RESULT = 1; // cualquiera -> lider: {numero de tarea, accuracy}
const int TAG_BEST = 2;   // lider -> resto de evaluacion: {K, es_definitivo}
const int TAG_REQ = 3;    // (dinamico) peticion de tarea: cualquiera -> lider
const int TAG_TASK = 4;   // (dinamico) respuesta del lider: {numero de tarea | -1 si no quedan}
const int CHUNK = 25;     // muestras que se clasifican entre dos consultas de mensajes

[[noreturn]] void abortAll(int rank, const std::string &msg)
{
    std::cerr << "[proceso " << rank << "] " << msg << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::exit(1);
}

bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    return loadMnist("train/train-images-idx3-ubyte", "train/train-labels-idx1-ubyte", train)
        && loadMnist("test/t10k-images-idx3-ubyte", "test/t10k-labels-idx1-ubyte", test);
}

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

// Accuracy del KNN con 'k' vecinos usando el fold 'fold' como validacion. Misma
// particion que CrossValidation::splitDataset del codigo secuencial. Si se pasa
// 'poll', se llama cada CHUNK muestras (para atender mensajes mientras se trabaja).
double evaluateFold(const std::vector<TSample> &data, unsigned folds, unsigned fold, int k,
                    const std::function<void()> &poll = nullptr)
{
    const std::size_t start = data.size() * fold / folds;
    const std::size_t end = data.size() * (fold + 1) / folds;

    std::vector<TSample> train;
    train.reserve(data.size() - (end - start));
    train.insert(train.end(), data.begin(), data.begin() + start);
    train.insert(train.end(), data.begin() + end, data.end());

    KNN knn(train, k);
    long long matches = 0;
    for (std::size_t i = start; i < end;)
    {
        const std::size_t chunkEnd = std::min(i + static_cast<std::size_t>(CHUNK), end);
        matches += countMatches(knn, data, i, chunkEnd);
        i = chunkEnd;
        if (poll) poll();
    }
    return matches / (double)(end - start);
}

// Barrera que NO consume CPU mientras espera. Un MPI_Barrier normal "gira" al
// 100 % de CPU esperando; si hay mas procesos que nucleos, quitaria tiempo a
// los que aun estan trabajando y falsearia las medidas.
void sleepyBarrier(MPI_Comm comm)
{
    MPI_Request req;
    MPI_Ibarrier(comm, &req);
    int done = 0;
    while (!done)
    {
        MPI_Test(&req, &done, MPI_STATUS_IGNORE);
        if (!done) usleep(1000);
    }
}

// -----------------------------------------------------------------------------
// Lleva la cuenta de los resultados del ajuste que van llegando (solo el lider).
// -----------------------------------------------------------------------------
class BestKTracker
{
public:
    BestKTracker(int minK, int numK, unsigned folds)
        : minK_(minK), folds_(folds), acc_(numK * folds, 0.0), got_(numK, 0), accK_(numK, 0.0) {}

    // Registra el resultado de la tarea (k, fold). Devuelve true si, con ello,
    // cambia el mejor K conocido.
    bool add(int task, double accuracy)
    {
        acc_[task] = accuracy;
        received_++;
        const int ki = task / static_cast<int>(folds_);
        if (++got_[ki] < static_cast<int>(folds_))
            return false; // este K aun no tiene todos sus folds

        // Media de los folds EN ORDEN: mismos calculos (y mismos bits) que el secuencial.
        double a = 0.0;
        for (unsigned f = 0; f < folds_; f++)
            a += acc_[ki * folds_ + f];
        a /= folds_;
        accK_[ki] = a;

        if (bestKi_ < 0 || a > accK_[bestKi_] || (a == accK_[bestKi_] && ki < bestKi_))
        {
            bestKi_ = ki;
            return true;
        }
        return false;
    }

    bool complete() const { return received_ == static_cast<int>(acc_.size()); }
    int bestK() const { return minK_ + bestKi_; }
    double accuracyOf(int ki) const { return accK_[ki]; }

private:
    int minK_;
    unsigned folds_;
    std::vector<double> acc_; // acc_[(k - minK) * folds + fold]
    std::vector<int> got_;    // folds recibidos de cada K
    std::vector<double> accK_;
    int received_ = 0;
    int bestKi_ = -1;
};

struct TimelineEntry
{
    double t;   // segundos desde el inicio
    int k;      // mejor K conocido en ese momento
    bool final; // true si el ajuste ya habia terminado
};

struct EvalStats
{
    int restarts = 0;       // veces que se descarto el trabajo por llegar un K distinto
    double compute = 0.0;   // segundos clasificando test (incluye el trabajo descartado)
    double wasted = 0.0;    // segundos de trabajo descartado
    double wait = 0.0;      // segundos esperando a que termine el ajuste
    int helpTasks = 0;      // (dinamico) tareas de ajuste hechas por este proceso de evaluacion
    double helpCompute = 0.0;
    std::vector<TimelineEntry> timeline; // solo el lider
};

// -----------------------------------------------------------------------------
// GRUPO DE AJUSTE, modo estatico: tareas repartidas ciclicamente de antemano.
// Devuelve los segundos de calculo y, en 'done', cuantas tareas ha hecho.
// -----------------------------------------------------------------------------
double runTuningStatic(const std::vector<TSample> &tuning, unsigned folds, int minK, int numTasks,
                       int myIndex, int nT, int leaderRank, int &done)
{
    std::vector<int> tasks;
    for (int task = myIndex; task < numTasks; task += nT) // reparto ciclico
        tasks.push_back(task);

    std::vector<std::array<double, 2>> msgs(tasks.size()); // buffers que deben seguir vivos hasta el Waitall
    std::vector<MPI_Request> reqs(tasks.size());
    double compute = 0.0;

    for (std::size_t idx = 0; idx < tasks.size(); idx++)
    {
        const int task = tasks[idx];
        const double t0 = MPI_Wtime();
        const double acc = evaluateFold(tuning, folds, task % folds, minK + task / folds);
        compute += MPI_Wtime() - t0;

        msgs[idx] = {static_cast<double>(task), acc};
        // Envio no bloqueante: sigo con la siguiente tarea sin esperar al lider.
        MPI_Isend(msgs[idx].data(), 2, MPI_DOUBLE, leaderRank, TAG_RESULT, MPI_COMM_WORLD, &reqs[idx]);
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    done = static_cast<int>(tasks.size());
    return compute;
}

// -----------------------------------------------------------------------------
// GRUPO DE AJUSTE, modo dinamico: pide tareas al lider hasta que le contesta -1.
// -----------------------------------------------------------------------------
double runTuningDynamic(const std::vector<TSample> &tuning, unsigned folds, int minK,
                        int leaderRank, int &done)
{
    std::deque<std::array<double, 2>> msgs;
    std::vector<MPI_Request> reqs;
    double compute = 0.0;
    done = 0;

    for (;;)
    {
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, leaderRank, TAG_REQ, MPI_COMM_WORLD);

        // Espera la respuesta durmiendo (el lider tarda a lo sumo unas decimas en atender).
        int task = -1, flag = 0;
        for (;;)
        {
            MPI_Iprobe(leaderRank, TAG_TASK, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (flag) break;
            usleep(1000);
        }
        MPI_Recv(&task, 1, MPI_INT, leaderRank, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (task < 0)
            break; // no quedan tareas

        const double t0 = MPI_Wtime();
        const double acc = evaluateFold(tuning, folds, task % folds, minK + task / folds);
        compute += MPI_Wtime() - t0;
        done++;

        msgs.push_back({static_cast<double>(task), acc});
        reqs.emplace_back();
        MPI_Isend(msgs.back().data(), 2, MPI_DOUBLE, leaderRank, TAG_RESULT, MPI_COMM_WORLD, &reqs.back());
    }
    MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
    return compute;
}

// -----------------------------------------------------------------------------
// GRUPO DE EVALUACION: clasifica begin..end con el mejor K conocido y reinicia
// si cambia. En modo dinamico, ayuda con el ajuste cuando se queda sin trabajo.
// -----------------------------------------------------------------------------
class Evaluator
{
public:
    struct Config
    {
        const std::vector<TSample> *train = nullptr;  // entrenamiento de la evaluacion final
        const std::vector<TSample> *test = nullptr;
        const std::vector<TSample> *tuning = nullptr; // datos del ajuste (solo para ayudar)
        std::size_t begin = 0, end = 0;               // bloque de test de este proceso
        int k0 = 1;                                   // K inicial supuesto
        int leaderRank = 0;
        bool isLeader = false;
        std::vector<int> workers;                     // resto del grupo (solo lo usa el lider)
        BestKTracker *tracker = nullptr;              // solo el lider
        double tStart = 0.0;
        bool dynamic = false;
        unsigned folds = 5;
        int minK = 1;
        int numTasks = 0;
    };

    Evaluator(const Config &c, EvalStats &stats) : c_(c), stats_(stats), kWanted_(c.k0) {}

    // Devuelve los aciertos del bloque con el K definitivo (en finalK).
    long long run(int &finalK)
    {
        int kUsed = -1;          // K con el que se esta clasificando de verdad
        long long matches = 0;
        std::size_t i = c_.begin;
        double workThisK = 0.0;  // trabajo hecho con kUsed (se pierde si hay que reiniciar)
        std::unique_ptr<KNN> knn;
        bool canHelp = c_.dynamic;

        for (;;)
        {
            if (kUsed != kWanted_) // (re)inicio: o es la primera vez, o ha llegado un K distinto
            {
                if (kUsed != -1)
                {
                    stats_.restarts++;
                    stats_.wasted += workThisK; // lo hecho con el K anterior se descarta
                }
                const double t0 = MPI_Wtime();
                kUsed = kWanted_;
                knn.reset(new KNN(*c_.train, kUsed));
                matches = 0;
                i = c_.begin;
                workThisK = MPI_Wtime() - t0;
                stats_.compute += workThisK;
            }

            if (i < c_.end) // quedan muestras: un trozo de CHUNK y luego miro mensajes
            {
                const double t0 = MPI_Wtime();
                const std::size_t chunkEnd = std::min(i + static_cast<std::size_t>(CHUNK), c_.end);
                matches += countMatches(*knn, *c_.test, i, chunkEnd);
                i = chunkEnd;
                const double dt = MPI_Wtime() - t0;
                workThisK += dt;
                stats_.compute += dt;
                poll();
                continue;
            }

            // Bloque terminado con kUsed.
            if (finalKnown_ && kWanted_ == kUsed)
                break; // el K usado es el definitivo: hecho

            // Aun puede cambiar el K. Modo dinamico: en vez de esperar, ayudo con el ajuste.
            if (canHelp)
            {
                const int task = fetchTask();
                if (task >= 0)
                {
                    helpWith(task);
                    continue; // puede haber cambiado el K mientras ayudaba
                }
                canHelp = false; // no quedan tareas
            }

            // Espero mensajes sin ocupar CPU.
            const double t0 = MPI_Wtime();
            while (kWanted_ == kUsed && !finalKnown_)
            {
                if (!poll())
                    usleep(1000);
            }
            stats_.wait += MPI_Wtime() - t0;
        }

        MPI_Waitall(static_cast<int>(outReq_.size()), outReq_.data(), MPI_STATUSES_IGNORE);
        finalK = kUsed;
        return matches;
    }

    // Barrera final. El lider sigue atendiendo peticiones de tarea mientras espera:
    // asi nadie se queda esperando una respuesta que no llega.
    void finalBarrier(bool serve)
    {
        if (!serve)
        {
            sleepyBarrier(MPI_COMM_WORLD);
            return;
        }
        MPI_Request req;
        MPI_Ibarrier(MPI_COMM_WORLD, &req);
        int done = 0;
        while (!done)
        {
            MPI_Test(&req, &done, MPI_STATUS_IGNORE);
            if (!done && !poll())
                usleep(1000);
        }
    }

private:
    // Envia el aviso (K, definitivo?) a todo el grupo de evaluacion (solo el lider).
    void announce(int k, bool fin)
    {
        for (int w : c_.workers)
        {
            outBuf_.push_back({k, fin ? 1 : 0});
            outReq_.emplace_back();
            MPI_Isend(outBuf_.back().data(), 2, MPI_INT, w, TAG_BEST, MPI_COMM_WORLD, &outReq_.back());
        }
    }

    // El lider registra un resultado de ajuste (llegado por mensaje o calculado por el).
    void onResult(int task, double accuracy)
    {
        const bool changed = c_.tracker->add(task, accuracy);
        if (c_.tracker->complete())
        {
            kWanted_ = c_.tracker->bestK();
            finalKnown_ = true;
            stats_.timeline.push_back({MPI_Wtime() - c_.tStart, kWanted_, true});
            announce(kWanted_, true);
        }
        else if (changed)
        {
            kWanted_ = c_.tracker->bestK();
            stats_.timeline.push_back({MPI_Wtime() - c_.tStart, kWanted_, false});
            announce(kWanted_, false);
        }
    }

    // Consulta mensajes sin bloquear. Devuelve true si ha procesado alguno.
    bool poll()
    {
        bool got = false;
        int flag = 0;
        if (c_.isLeader)
        {
            for (;;) // resultados del ajuste
            {
                MPI_Status st;
                MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &flag, &st);
                if (!flag) break;
                double msg[2];
                MPI_Recv(msg, 2, MPI_DOUBLE, st.MPI_SOURCE, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                got = true;
                onResult(static_cast<int>(msg[0]), msg[1]);
            }
            if (c_.dynamic)
            {
                for (;;) // peticiones de tarea: SIEMPRE se contestan
                {
                    MPI_Status st;
                    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQ, MPI_COMM_WORLD, &flag, &st);
                    if (!flag) break;
                    int dummy;
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, TAG_REQ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    got = true;
                    int task = nextTask_ < c_.numTasks ? nextTask_++ : -1;
                    MPI_Send(&task, 1, MPI_INT, st.MPI_SOURCE, TAG_TASK, MPI_COMM_WORLD);
                }
            }
        }
        else
        {
            for (;;) // avisos del lider
            {
                MPI_Iprobe(c_.leaderRank, TAG_BEST, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
                if (!flag) break;
                int msg[2];
                MPI_Recv(msg, 2, MPI_INT, c_.leaderRank, TAG_BEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                got = true;
                kWanted_ = msg[0];
                finalKnown_ = (msg[1] != 0);
            }
        }
        return got;
    }

    // Modo dinamico: obtiene una tarea de ajuste (-1 si no quedan).
    int fetchTask()
    {
        if (c_.isLeader)
            return nextTask_ < c_.numTasks ? nextTask_++ : -1;

        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, c_.leaderRank, TAG_REQ, MPI_COMM_WORLD);
        for (;;) // la respuesta SIEMPRE llega; mientras tanto sigo atendiendo avisos de K
        {
            int flag = 0;
            MPI_Iprobe(c_.leaderRank, TAG_TASK, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
            if (flag)
            {
                int task;
                MPI_Recv(&task, 1, MPI_INT, c_.leaderRank, TAG_TASK, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                return task;
            }
            if (!poll())
                usleep(1000);
        }
    }

    // Calcula una tarea de ajuste y entrega el resultado al lider (atendiendo mensajes mientras tanto).
    void helpWith(int task)
    {
        const double t0 = MPI_Wtime();
        const double acc = evaluateFold(*c_.tuning, c_.folds, task % c_.folds, c_.minK + task / c_.folds,
                                        [this]() { poll(); });
        stats_.helpCompute += MPI_Wtime() - t0;
        stats_.helpTasks++;

        if (c_.isLeader)
        {
            onResult(task, acc);
        }
        else
        {
            outRes_.push_back({static_cast<double>(task), acc});
            outReq_.emplace_back();
            MPI_Isend(outRes_.back().data(), 2, MPI_DOUBLE, c_.leaderRank, TAG_RESULT, MPI_COMM_WORLD,
                      &outReq_.back());
        }
    }

    Config c_;
    EvalStats &stats_;
    int kWanted_;            // el K que ahora mismo se cree el mejor
    bool finalKnown_ = false; // true cuando el ajuste ha terminado y kWanted_ es el definitivo
    int nextTask_ = 0;       // (lider, dinamico) siguiente tarea de ajuste por repartir

    std::deque<std::array<int, 2>> outBuf_;      // buffers de envios: deben seguir vivos hasta el Waitall
    std::deque<std::array<double, 2>> outRes_;
    std::vector<MPI_Request> outReq_;
};

const int NFIELDS = 9; // campos de metricas por proceso (ver main)

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Para gprof (binarios compilados con -pg): cada proceso escribe su propio fichero gmon_rank<N>.<pid>
    // en vez de pisarse todos en gmon.out. En los binarios normales esta variable no se usa.
    { char gp[32]; std::snprintf(gp, sizeof gp, "gmon_rank%d", rank); setenv("GMON_OUT_PREFIX", gp, 1); }
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2)
        abortAll(rank, "Esta version necesita al menos 2 procesos (un grupo de ajuste y otro de evaluacion)");

    // ---- Parametros ---------------------------------------------------------
    const unsigned minK = 1;
    unsigned maxK = 10, folds = 5;
    std::size_t tuningSamples = 20000;
    std::size_t testSamples = std::numeric_limits<std::size_t>::max();
    std::size_t finalTrainSamples = std::numeric_limits<std::size_t>::max();
    int evalProcs = 0;       // 0 = automatico
    unsigned initialK = minK;
    unsigned mode = 0;       // 0 = estatico (paso 2), 1 = dinamico (paso 3)
    try
    {
        if (argc > 1) maxK = std::stoul(argv[1]);
        if (argc > 2) folds = std::stoul(argv[2]);
        if (argc > 3) tuningSamples = std::stoul(argv[3]);
        if (argc > 4) testSamples = std::stoul(argv[4]);
        if (argc > 5) finalTrainSamples = std::stoul(argv[5]);
        if (argc > 6) evalProcs = std::stoi(argv[6]);
        if (argc > 7) initialK = std::stoul(argv[7]);
        if (argc > 8) mode = std::stoul(argv[8]);
    }
    catch (const std::exception &)
    {
        abortAll(rank, "Uso: knn_mpi_spec [maxK] [folds] [muestras_ajuste] [muestras_test] "
                       "[entrenamiento_final] [procesos_evaluacion] [k_inicial] [modo 0|1]");
    }
    if (maxK <= minK || folds < 2)
        abortAll(rank, "Se necesita maxK > 1 y folds >= 2");
    if (initialK < minK || initialK > maxK)
        abortAll(rank, "k_inicial debe estar entre 1 y maxK");
    if (mode > 1)
        abortAll(rank, "modo debe ser 0 (estatico) o 1 (dinamico)");
    const bool dynamic = (mode == 1);

    sleepyBarrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();

    // ---- Carga del dataset (cada proceso, sin comunicacion) ------------------
    std::vector<TSample> train, test;
    if (!loadDataset(train, test))
        abortAll(rank, "No se pudo leer MNIST: faltan las carpetas train/ y test/ en el directorio actual");
    const double tLoad = MPI_Wtime() - tStart;

    const std::size_t nTuning = std::min(train.size(), tuningSamples);
    if (folds > nTuning)
        abortAll(rank, "Hay menos muestras de ajuste que folds");
    const std::vector<TSample> tuning(train.begin(), train.begin() + nTuning);

    const std::size_t nTest = std::min(test.size(), testSamples);
    const std::size_t nFinalTrain = std::min(train.size(), finalTrainSamples);
    std::vector<TSample> reducedTrain;
    const std::vector<TSample> *finalTrain = &train;
    if (nFinalTrain < train.size())
    {
        reducedTrain.assign(train.begin(), train.begin() + nFinalTrain);
        finalTrain = &reducedTrain;
    }

    // ---- Reparto en dos grupos ----------------------------------------------
    const int numK = maxK - minK + 1;
    const int numTasks = numK * folds;

    // Trabajo de cada fase en "distancias calculadas": ajuste = numK * n^2 * (folds-1)/folds;
    // evaluacion final = test * entrenamiento_final.
    const double tuningWork = double(numK) * nTuning * nTuning * (folds - 1) / folds;
    const double finalWork = double(nTest) * nFinalTrain;
    int nE = evalProcs > 0 ? evalProcs : static_cast<int>(std::lround(size * finalWork / (finalWork + tuningWork)));
    nE = std::max(1, std::min(nE, size - 1));
    const int nT = size - nE;

    const bool isEval = rank >= nT;
    const int leaderRank = nT; // primer proceso del grupo de evaluacion
    const bool isLeader = (rank == leaderRank);

    // Un comunicador por grupo: las colectivas del grupo de evaluacion solo lo implican a el.
    MPI_Comm groupComm;
    MPI_Comm_split(MPI_COMM_WORLD, isEval ? 1 : 0, rank, &groupComm);

    // ---- Cada proceso hace lo suyo -------------------------------------------
    EvalStats stats;
    double tuneCompute = 0.0; // segundos en tareas de ajuste (grupo de ajuste)
    int tuneTasks = 0;
    long long matches = 0;
    int finalK = -1;
    std::unique_ptr<BestKTracker> tracker;
    std::unique_ptr<Evaluator> evaluator;

    if (!isEval)
    {
        tuneCompute = dynamic ? runTuningDynamic(tuning, folds, minK, leaderRank, tuneTasks)
                              : runTuningStatic(tuning, folds, minK, numTasks, rank, nT, leaderRank, tuneTasks);
    }
    else
    {
        const int er = rank - nT; // indice dentro del grupo de evaluacion

        Evaluator::Config cfg;
        cfg.train = finalTrain;
        cfg.test = &test;
        cfg.tuning = &tuning;
        cfg.begin = nTest * static_cast<std::size_t>(er) / nE;
        cfg.end = nTest * static_cast<std::size_t>(er + 1) / nE;
        cfg.k0 = initialK;
        cfg.leaderRank = leaderRank;
        cfg.isLeader = isLeader;
        cfg.tStart = tStart;
        cfg.dynamic = dynamic;
        cfg.folds = folds;
        cfg.minK = minK;
        cfg.numTasks = numTasks;
        if (isLeader)
        {
            tracker.reset(new BestKTracker(minK, numK, folds));
            cfg.tracker = tracker.get();
            for (int r = nT + 1; r < size; r++)
                cfg.workers.push_back(r);
        }

        evaluator.reset(new Evaluator(cfg, stats));
        matches = evaluator->run(finalK);
        tuneCompute = stats.helpCompute;
        tuneTasks = stats.helpTasks;
    }
    const double tDone = MPI_Wtime() - tStart; // cuando termina ESTE proceso su trabajo

    // Barrera final. Los procesos de evaluacion siguen atendiendo peticiones hasta que todos llegan.
    if (isEval)
        evaluator->finalBarrier(isLeader);
    else
        sleepyBarrier(MPI_COMM_WORLD);

    long long totalMatches = 0;
    if (isEval)
        MPI_Reduce(&matches, &totalMatches, 1, MPI_LONG_LONG, MPI_SUM, 0, groupComm); // 0 = lider

    // ---- Recogida de metricas en el lider -----------------------------------
    // {es_evaluacion, fin, calculo_evaluacion, espera, descartado, reinicios, carga, tareas_ajuste, calculo_ajuste}
    const double mine[NFIELDS] = {isEval ? 1.0 : 0.0, tDone, isEval ? stats.compute : 0.0, stats.wait,
                                  stats.wasted, static_cast<double>(stats.restarts), tLoad,
                                  static_cast<double>(tuneTasks), tuneCompute};
    std::vector<double> all(NFIELDS * size);
    MPI_Gather(mine, NFIELDS, MPI_DOUBLE, all.data(), NFIELDS, MPI_DOUBLE, leaderRank, MPI_COMM_WORLD);

    if (isLeader)
    {
        double total = 0.0, maxLoad = 0.0, wastedSum = 0.0;
        for (int r = 0; r < size; r++)
        {
            const double *v = &all[NFIELDS * r];
            total = std::max(total, v[1]);
            maxLoad = std::max(maxLoad, v[6]);
            if (v[0] > 0.5) wastedSum += v[4];
        }

        std::cout << "Procesos MPI: " << size << " (ajuste: " << nT << ", evaluacion: " << nE << ") | modo "
                  << (dynamic ? "dinamico" : "estatico") << " | K en [" << minK << ", " << maxK << "], " << folds
                  << " folds, " << nTuning << " muestras de ajuste, " << nTest << " de test, " << nFinalTrain
                  << " de entrenamiento final, K inicial " << initialK << std::endl;
        // Mismo formato que el secuencial y el paso 1 (6 cifras significativas).
        for (int ki = 0; ki < numK; ki++)
            std::cout << "K: " << minK + ki << " Accuracy: " << tracker->accuracyOf(ki) * 100 << "%" << std::endl;
        std::cout << "Best k: " << finalK << std::endl;
        std::cout << "Test accuracy: " << totalMatches / (double)nTest << std::endl;

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Evolucion del mejor K conocido (segundos desde el inicio):" << std::endl;
        std::cout << "  t=" << 0.0 << "  K inicial supuesto = " << initialK << std::endl;
        for (const TimelineEntry &e : stats.timeline)
            std::cout << "  t=" << e.t << "  " << (e.final ? "ajuste completo, K definitivo = " : "nuevo mejor K = ")
                      << e.k << std::endl;
        std::cout << "Reinicios de la evaluacion (lider): " << stats.restarts
                  << ", trabajo descartado (suma de evaluadores): " << wastedSum << " s" << std::endl;

        std::cout << "Por proceso (segundos):" << std::endl;
        for (int r = 0; r < size; r++)
        {
            const double *v = &all[NFIELDS * r];
            std::cout << "  proceso " << r << (v[0] > 0.5 ? " [evaluacion]" : " [ajuste]    ")
                      << "  termina a t=" << v[1];
            if (v[0] > 0.5)
            {
                std::cout << "  evaluacion=" << v[2] << " (descartado=" << v[4] << ")  espera=" << v[3]
                          << "  reinicios=" << static_cast<int>(v[5]);
                if (dynamic)
                    std::cout << "  ayuda al ajuste: " << static_cast<int>(v[7]) << " tareas, " << v[8] << " s";
            }
            else
            {
                std::cout << "  tareas=" << static_cast<int>(v[7]) << "  calculo=" << v[8];
            }
            std::cout << std::endl;
        }
        std::cout << "Carga del dataset (replicada, maximo): " << maxLoad << std::endl;
        std::cout << "Total elapsed time " << total << " seconds" << std::endl;

        // Linea legible por maquina (la recoge run_all.sh). Mismas columnas que knn_mpi;
        // aqui no hay tune_s / final_s / comm_s (las fases se solapan) pero si grupos y reinicios.
        const std::string version = dynamic ? "knn_mpi_dyn" : "knn_mpi_spec";
        char csv[512];
        std::snprintf(csv, sizeof csv, "CSV,%s,%d,%zu,%zu,%zu,%u,%u,%d,%.6f,%.6f,%.6f,,,,%d,%d,%d,%.6f",
                      version.c_str(), size, nTuning, nTest, nFinalTrain, maxK, folds, finalK,
                      totalMatches / (double)nTest, total, maxLoad, nT, nE, stats.restarts, wastedSum);
        std::cout << csv << std::endl;
    }

    MPI_Comm_free(&groupComm);
    MPI_Finalize();
    return 0;
}
