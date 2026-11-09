// defect_detection_mpi.cc
//
// Version paralela (MPI) de defect_detection.cc (Practica 2). Hace EXACTAMENTE
// el mismo calculo (4 filtros 3x3 + 2 combinaciones + marcado) y produce las
// mismas imagenes de salida, byte a byte, que la version secuencial.
//
// Grafo de dependencias del calculo (por cada repeticion):
//
//                       original
//          ┌──────────┬────┴─────┬──────────┐
//        blur      sharpen    sobel-X    sobel-Y      <- 4 tareas independientes
//          └────┬─────┘          └────┬─────┘
//             suma                pitagoras           <- dependen de 2 filtros
//          (realzada)        (bordes + sospechosos)
//               └─────────┬──────────┘
//                      marcado                        <- depende de ambas ramas
//
// Ademas, DENTRO de cada tarea cada pixel es independiente (solo lee una
// vecindad 3x3 de la entrada) -> paralelismo de datos.
//
// Se implementan dos estrategias, seleccionables por linea de comandos:
//
//   datos  (por defecto) Descomposicion de DOMINIO por bloques de filas.
//          Cada proceso recibe un bloque de filas (MPI_Scatterv), intercambia
//          con sus vecinos la fila frontera ("halo", MPI_Sendrecv), calcula
//          TODA la cadena (4 filtros + combinaciones + marcado) sobre sus
//          filas, y el proceso 0 recoge los resultados (MPI_Gatherv) y el
//          numero de pixeles sospechosos (MPI_Reduce). Escala con P.
//
//   tareas Descomposicion FUNCIONAL siguiendo el grafo de arriba: cada filtro
//          es una tarea asignada a un proceso (maximo 4 procesos utiles).
//          Todos reciben la imagen completa (MPI_Bcast) y los resultados
//          intermedios viajan por las aristas del grafo (MPI_Isend/MPI_Recv).
//          Su speedup esta acotado por el numero de tareas (4) y por la parte
//          que queda secuencial (combinaciones y marcado).
//
// Modelo: SPMD (un unico programa, el comportamiento depende del rango) con
// estructura maestro-esclavo: solo el proceso 0 lee y escribe ficheros, asi
// que los demas nodos solo necesitan el ejecutable (no las imagenes).
//
// Parametro N (opcional, por defecto 1): igual que en la version secuencial,
// repite N veces el procesado de la imagen. En cada repeticion se repite
// TAMBIEN el reparto y la recogida de datos (como si llegaran N imagenes), para
// que el coste de comunicacion no quede amortizado artificialmente.
//
// Compilacion:  mpic++ -O2 -std=c++17 -o defect_detection_mpi defect_detection_mpi.cc
// Ejecucion:    mpirun -n <P> [-hostfile hosts.txt] ./defect_detection_mpi
//                   entradas/<imagen>.pgm [N=1] [datos|tareas]
//
// La ULTIMA linea de stdout (solo la imprime el proceso 0) es una linea CSV:
//   estrategia,procesos,lectura_ms,paralelo_ms,computo_max_ms,computo_medio_ms,
//   comunicacion_max_ms,comunicacion_media_ms,escritura_ms,total_ms

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Imagenes y E/S (identico a la version secuencial)
// ---------------------------------------------------------------------------

struct Image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;

    uint8_t at(int x, int y) const {
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return pixels[static_cast<size_t>(y) * width + x];
    }
};

Image readPGM(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("No se pudo abrir la imagen: " + path);
    }

    std::string magic;
    file >> magic;
    if (magic != "P5" && magic != "P2") {
        throw std::runtime_error("Formato no soportado (se espera PGM P5 o P2): " + magic);
    }

    auto skipComments = [&file]() {
        while (file.peek() == '#') {
            std::string line;
            std::getline(file, line);
        }
    };

    int width = 0, height = 0, maxVal = 0;
    skipComments();
    file >> width;
    skipComments();
    file >> height;
    skipComments();
    file >> maxVal;
    file.get();

    Image img;
    img.width = width;
    img.height = height;
    img.pixels.resize(static_cast<size_t>(width) * height);

    if (magic == "P5") {
        file.read(reinterpret_cast<char*>(img.pixels.data()),
                  static_cast<std::streamsize>(img.pixels.size()));
    } else {
        for (auto& p : img.pixels) {
            int v;
            file >> v;
            p = static_cast<uint8_t>(v);
        }
    }
    return img;
}

void writePGM(const std::string& path, int w, int h, const std::vector<uint8_t>& px) {
    std::ofstream file(path, std::ios::binary);
    file << "P5\n" << w << " " << h << "\n255\n";
    file.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
}

void writePPM(const std::string& path, int w, int h, const std::vector<uint8_t>& rgb) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << w << " " << h << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

// ---------------------------------------------------------------------------
// Nucleos de calculo. Son las mismas operaciones que la version secuencial,
// pero trabajan sobre un rango de filas / un trozo de vector para que cada
// proceso pueda calcular solo su parte.
// ---------------------------------------------------------------------------

static const float blurK[3][3] = {
    {1.f / 16, 2.f / 16, 1.f / 16},
    {2.f / 16, 4.f / 16, 2.f / 16},
    {1.f / 16, 2.f / 16, 1.f / 16}};
static const float sharpenK[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
static const float sobelXK[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
static const float sobelYK[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

const int UMBRAL = 220;
const double UMBRAL_DENSIDAD = 3.75;
const float alpha = 0.65f;

// Convoluciona las filas [yIni, yFin) de src y deja el resultado en out
// (fila yIni de src -> fila 0 de out). Fuera de la imagen se replica el borde.
void convolve3x3(const Image& src, const float k[3][3], int yIni, int yFin, float* out) {
    const int w = src.width;
    for (int y = yIni; y < yFin; ++y) {
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += k[ky + 1][kx + 1] * src.at(x + kx, y + ky);
                }
            }
            out[static_cast<size_t>(y - yIni) * w + x] = sum;
        }
    }
}

void clampToU8(const float* in, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = static_cast<uint8_t>(std::clamp(in[i], 0.0f, 255.0f));
    }
}

// Combinacion 1: blur + sharpen -> imagen realzada
void combinarSuma(const uint8_t* blur, const uint8_t* sharp, uint8_t* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        int avg = (blur[i] + sharp[i]) / 2;
        out[i] = static_cast<uint8_t>(std::clamp(avg, 0, 255));
    }
}

// Combinacion 2: sobel-X + sobel-Y -> mapa de bordes. Devuelve cuantos
// pixeles superan el umbral y los marca en 'sosp'.
long combinarPitagoras(const float* gx, const float* gy, uint8_t* edges, uint8_t* sosp, size_t n) {
    long cuenta = 0;
    for (size_t i = 0; i < n; ++i) {
        const float mag = std::sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
        const uint8_t value = static_cast<uint8_t>(std::clamp(mag, 0.0f, 255.0f));
        edges[i] = value;
        sosp[i] = value > UMBRAL;
        cuenta += sosp[i];
    }
    return cuenta;
}

// Imagen para revision humana: realzada en gris + zonas sospechosas en rojo.
void marcar(const uint8_t* enh, const uint8_t* sosp, uint8_t* rgb, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        const float gray = enh[i];
        uint8_t r, g, b;
        if (sosp[i]) {
            r = static_cast<uint8_t>(gray * (1 - alpha) + 255 * alpha);
            g = static_cast<uint8_t>(gray * (1 - alpha));
            b = static_cast<uint8_t>(gray * (1 - alpha));
        } else {
            r = g = b = static_cast<uint8_t>(gray);
        }
        rgb[i * 3 + 0] = r;
        rgb[i * 3 + 1] = g;
        rgb[i * 3 + 2] = b;
    }
}

// ---------------------------------------------------------------------------
// Medida de tiempos por proceso: computo vs. comunicacion (+ espera).
// ---------------------------------------------------------------------------

struct Tiempos {
    double computo = 0;       // segundos
    double comunicacion = 0;  // segundos (incluye esperas/sincronizacion)
};

// Resultado que necesita el proceso 0 para escribir las salidas.
struct Resultado {
    std::vector<uint8_t> edges;  // W*H
    std::vector<uint8_t> rgb;    // W*H*3
    long sospechosos = 0;
};

enum Tag { TAG_HALO_SUBE = 1, TAG_HALO_BAJA, TAG_SHARP, TAG_GY, TAG_EDGES, TAG_SOSP, TAG_CUENTA };

// ---------------------------------------------------------------------------
// Estrategia 1: descomposicion de datos por bloques de filas
// ---------------------------------------------------------------------------
//
//   P0 ─┬─ Scatterv(filas) ─► P0..P(p-1)
//       │  Sendrecv(halo) entre vecinos  Pi <-> Pi+1
//       │  [cada Pi: 4 filtros + suma + pitagoras + marcado de SUS filas]
//       │  Reduce(SUM sospechosos) ─► P0
//       └─ Gatherv(bordes) y Gatherv(rgb) ─► P0
//
class EstrategiaDatos {
public:
    EstrategiaDatos(int W, int H, int rank, int P) : W_(W), rank_(rank), P_(P) {
        // Reparto equilibrado: las primeras H%P filas extra van a los
        // primeros procesos (diferencia maxima de 1 fila entre procesos).
        cnt_.resize(P);
        dsp_.resize(P);
        cnt3_.resize(P);
        dsp3_.resize(P);
        const int base = H / P, resto = H % P;
        int fila = 0;
        for (int r = 0; r < P; ++r) {
            const int filas = base + (r < resto ? 1 : 0);
            cnt_[r] = filas * W;
            dsp_[r] = fila * W;
            cnt3_[r] = cnt_[r] * 3;
            dsp3_[r] = dsp_[r] * 3;
            fila += filas;
        }
        filas_ = cnt_[rank] / W;
        n_ = static_cast<size_t>(filas_) * W;

        // Bloque local con una fila de halo arriba y otra abajo.
        local_.width = W;
        local_.height = filas_ + 2;
        local_.pixels.resize(static_cast<size_t>(filas_ + 2) * W);

        tmp_.resize(n_);
        gx_.resize(n_);
        gy_.resize(n_);
        blur_.resize(n_);
        sharp_.resize(n_);
        enh_.resize(n_);
        edges_.resize(n_);
        sosp_.resize(n_);
        rgb_.resize(n_ * 3);
    }

    void repeticion(const Image& original, Resultado& res, Tiempos& t) {
        double t0 = MPI_Wtime();

        // 1) Reparto de bloques de filas (dejando libre la fila 0 = halo sup.)
        uint8_t* haloSup = local_.pixels.data();
        uint8_t* primera = haloSup + W_;
        uint8_t* ultima = primera + static_cast<size_t>(filas_ - 1) * W_;
        uint8_t* haloInf = ultima + W_;
        MPI_Scatterv(rank_ == 0 ? original.pixels.data() : nullptr, cnt_.data(), dsp_.data(),
                     MPI_UNSIGNED_CHAR, primera, cnt_[rank_], MPI_UNSIGNED_CHAR, 0,
                     MPI_COMM_WORLD);

        // 2) Intercambio de halos con los vecinos (el kernel 3x3 necesita la
        //    fila de encima y la de debajo del bloque). En los extremos de la
        //    imagen el vecino es MPI_PROC_NULL (la llamada no hace nada).
        const int arriba = rank_ > 0 ? rank_ - 1 : MPI_PROC_NULL;
        const int abajo = rank_ < P_ - 1 ? rank_ + 1 : MPI_PROC_NULL;
        MPI_Sendrecv(primera, W_, MPI_UNSIGNED_CHAR, arriba, TAG_HALO_SUBE,
                     haloInf, W_, MPI_UNSIGNED_CHAR, abajo, TAG_HALO_SUBE,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(ultima, W_, MPI_UNSIGNED_CHAR, abajo, TAG_HALO_BAJA,
                     haloSup, W_, MPI_UNSIGNED_CHAR, arriba, TAG_HALO_BAJA,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double t1 = MPI_Wtime();
        t.comunicacion += t1 - t0;

        // Borde real de la imagen: replicar la fila (= clamp de la version
        // secuencial), asi el resultado es identico.
        if (arriba == MPI_PROC_NULL) std::memcpy(haloSup, primera, W_);
        if (abajo == MPI_PROC_NULL) std::memcpy(haloInf, ultima, W_);

        // 3) Computo local: toda la cadena sobre las filas 1..filas_ del bloque
        convolve3x3(local_, blurK, 1, filas_ + 1, tmp_.data());
        clampToU8(tmp_.data(), blur_.data(), n_);
        convolve3x3(local_, sharpenK, 1, filas_ + 1, tmp_.data());
        clampToU8(tmp_.data(), sharp_.data(), n_);
        convolve3x3(local_, sobelXK, 1, filas_ + 1, gx_.data());
        convolve3x3(local_, sobelYK, 1, filas_ + 1, gy_.data());

        combinarSuma(blur_.data(), sharp_.data(), enh_.data(), n_);
        long cuenta = combinarPitagoras(gx_.data(), gy_.data(), edges_.data(), sosp_.data(), n_);
        marcar(enh_.data(), sosp_.data(), rgb_.data(), n_);
        double t2 = MPI_Wtime();
        t.computo += t2 - t1;

        // 4) Recogida: solo se envia lo que el proceso 0 necesita escribir
        //    (bordes y rgb), no los resultados intermedios.
        MPI_Reduce(&cuenta, &res.sospechosos, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Gatherv(edges_.data(), cnt_[rank_], MPI_UNSIGNED_CHAR,
                    rank_ == 0 ? res.edges.data() : nullptr, cnt_.data(), dsp_.data(),
                    MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        MPI_Gatherv(rgb_.data(), cnt3_[rank_], MPI_UNSIGNED_CHAR,
                    rank_ == 0 ? res.rgb.data() : nullptr, cnt3_.data(), dsp3_.data(),
                    MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        t.comunicacion += MPI_Wtime() - t2;
    }

private:
    int W_, rank_, P_;
    int filas_ = 0;
    size_t n_ = 0;
    std::vector<int> cnt_, dsp_, cnt3_, dsp3_;
    Image local_;
    std::vector<float> tmp_, gx_, gy_;
    std::vector<uint8_t> blur_, sharp_, enh_, edges_, sosp_, rgb_;
};

// ---------------------------------------------------------------------------
// Estrategia 2: descomposicion funcional (una tarea = un filtro)
// ---------------------------------------------------------------------------
//
// Asignacion de tareas a procesos (Q = procesos utiles = min(P, 4)):
//   blur -> 0, sharpen -> 1%Q, sobel-X -> 2%Q, sobel-Y -> 3%Q
//   suma -> dueno de blur (0), pitagoras -> dueno de sobel-X, marcado -> 0
// Los datos viajan solo por las aristas del grafo que unen procesos
// distintos. Con Q = 4:
//
//   P0: Bcast ─► blur ──────────┐ suma ─────────────┐ marcado
//   P1: Bcast ─► sharpen ─Isend─┘                   │
//   P2: Bcast ─► sobel-X ─────────┐ pitagoras ─Isend┘
//   P3: Bcast ─► sobel-Y ─Isend───┘
//
// Los procesos con rango >= 4 no tienen tarea (quedan fuera del
// comunicador 'comm_' creado con MPI_Comm_split).
//
class EstrategiaTareas {
public:
    EstrategiaTareas(int W, int H, int rank, int P) : rank_(rank) {
        const int Q = std::min(P, 4);
        activo_ = rank < Q;
        MPI_Comm_split(MPI_COMM_WORLD, activo_ ? 0 : MPI_UNDEFINED, rank, &comm_);
        if (!activo_) return;

        dBlur_ = 0;
        dSharp_ = 1 % Q;
        dSobelX_ = 2 % Q;
        dSobelY_ = 3 % Q;

        n_ = static_cast<size_t>(W) * H;
        img_.width = W;
        img_.height = H;
        img_.pixels.resize(n_);
        tmp_.resize(n_);
        if (rank_ == dBlur_) blur_.resize(n_);
        if (rank_ == dSharp_ || rank_ == 0) sharp_.resize(n_);
        if (rank_ == dSobelX_) gx_.resize(n_);
        if (rank_ == dSobelY_ || rank_ == dSobelX_) gy_.resize(n_);
        if (rank_ == 0) enh_.resize(n_);
        if (rank_ == dSobelX_ || rank_ == 0) {
            edges_.resize(n_);
            sosp_.resize(n_);
        }
    }

    ~EstrategiaTareas() {
        if (comm_ != MPI_COMM_NULL) MPI_Comm_free(&comm_);
    }

    void repeticion(const Image& original, Resultado& res, Tiempos& t) {
        if (!activo_) return;
        const int H = img_.height;
        std::vector<MPI_Request> reqs;

        // 1) Todos los procesos con tarea necesitan la imagen entera.
        double t0 = MPI_Wtime();
        if (rank_ == 0) std::memcpy(img_.pixels.data(), original.pixels.data(), n_);
        MPI_Bcast(img_.pixels.data(), static_cast<int>(n_), MPI_UNSIGNED_CHAR, 0, comm_);
        double t1 = MPI_Wtime();
        t.comunicacion += t1 - t0;

        // 2) Cada proceso calcula SUS filtros sobre la imagen completa.
        if (rank_ == dBlur_) {
            convolve3x3(img_, blurK, 0, H, tmp_.data());
            clampToU8(tmp_.data(), blur_.data(), n_);
        }
        if (rank_ == dSharp_) {
            convolve3x3(img_, sharpenK, 0, H, tmp_.data());
            clampToU8(tmp_.data(), sharp_.data(), n_);
        }
        if (rank_ == dSobelX_) convolve3x3(img_, sobelXK, 0, H, gx_.data());
        if (rank_ == dSobelY_) convolve3x3(img_, sobelYK, 0, H, gy_.data());
        double t2 = MPI_Wtime();
        t.computo += t2 - t1;

        // 3) Aristas del grafo entre procesos distintos.
        //    Envios no bloqueantes (Isend) para que ningun proceso se quede
        //    bloqueado enviando mientras otro espera; se completan con Waitall.
        //    sharpen -> suma (proceso 0)
        if (dSharp_ != 0) {
            if (rank_ == dSharp_) isend(sharp_.data(), n_, MPI_UNSIGNED_CHAR, 0, TAG_SHARP, reqs);
            if (rank_ == 0)
                MPI_Recv(sharp_.data(), static_cast<int>(n_), MPI_UNSIGNED_CHAR, dSharp_,
                         TAG_SHARP, comm_, MPI_STATUS_IGNORE);
        }
        //    sobel-Y -> pitagoras (dueno de sobel-X)
        if (dSobelY_ != dSobelX_) {
            if (rank_ == dSobelY_) isend(gy_.data(), n_, MPI_FLOAT, dSobelX_, TAG_GY, reqs);
            if (rank_ == dSobelX_)
                MPI_Recv(gy_.data(), static_cast<int>(n_), MPI_FLOAT, dSobelY_, TAG_GY, comm_,
                         MPI_STATUS_IGNORE);
        }
        double t3 = MPI_Wtime();
        t.comunicacion += t3 - t2;

        // 4) Combinaciones (cada una en el proceso que ya tiene sus entradas)
        long cuenta = 0;
        if (rank_ == 0) combinarSuma(blur_.data(), sharp_.data(), enh_.data(), n_);
        if (rank_ == dSobelX_)
            cuenta = combinarPitagoras(gx_.data(), gy_.data(), edges_.data(), sosp_.data(), n_);
        double t4 = MPI_Wtime();
        t.computo += t4 - t3;

        //    pitagoras -> marcado (proceso 0)
        if (dSobelX_ != 0) {
            if (rank_ == dSobelX_) {
                isend(edges_.data(), n_, MPI_UNSIGNED_CHAR, 0, TAG_EDGES, reqs);
                isend(sosp_.data(), n_, MPI_UNSIGNED_CHAR, 0, TAG_SOSP, reqs);
                isend(&cuenta, 1, MPI_LONG, 0, TAG_CUENTA, reqs);
            }
            if (rank_ == 0) {
                MPI_Recv(edges_.data(), static_cast<int>(n_), MPI_UNSIGNED_CHAR, dSobelX_,
                         TAG_EDGES, comm_, MPI_STATUS_IGNORE);
                MPI_Recv(sosp_.data(), static_cast<int>(n_), MPI_UNSIGNED_CHAR, dSobelX_,
                         TAG_SOSP, comm_, MPI_STATUS_IGNORE);
                MPI_Recv(&cuenta, 1, MPI_LONG, dSobelX_, TAG_CUENTA, comm_, MPI_STATUS_IGNORE);
            }
        }
        double t5 = MPI_Wtime();
        t.comunicacion += t5 - t4;

        // 5) Marcado (parte secuencial, solo proceso 0)
        if (rank_ == 0) {
            marcar(enh_.data(), sosp_.data(), res.rgb.data(), n_);
            std::memcpy(res.edges.data(), edges_.data(), n_);
            res.sospechosos = cuenta;
        }
        double t6 = MPI_Wtime();
        t.computo += t6 - t5;

        // Los buffers enviados con Isend no se pueden reutilizar hasta aqui.
        MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        t.comunicacion += MPI_Wtime() - t6;
    }

private:
    void isend(const void* buf, size_t n, MPI_Datatype tipo, int dest, int tag,
               std::vector<MPI_Request>& reqs) {
        reqs.emplace_back();
        MPI_Isend(buf, static_cast<int>(n), tipo, dest, tag, comm_, &reqs.back());
    }

    int rank_;
    bool activo_ = false;
    MPI_Comm comm_ = MPI_COMM_NULL;
    int dBlur_ = 0, dSharp_ = 0, dSobelX_ = 0, dSobelY_ = 0;
    size_t n_ = 0;
    Image img_;
    std::vector<float> tmp_, gx_, gy_;
    std::vector<uint8_t> blur_, sharp_, enh_, edges_, sosp_;
};

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2) {
        if (rank == 0)
            std::cerr << "Uso: mpirun -n <P> " << argv[0]
                      << " entradas/<nombre>.pgm [N=1] [datos|tareas]\n";
        MPI_Finalize();
        return 1;
    }
    const std::filesystem::path inputPath = argv[1];
    const int N = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;
    const std::string estrategia = argc > 3 ? argv[3] : "datos";
    if (estrategia != "datos" && estrategia != "tareas") {
        if (rank == 0) std::cerr << "Estrategia desconocida: " << estrategia << "\n";
        MPI_Finalize();
        return 1;
    }

    // --- Lectura: solo el proceso 0 accede al fichero ---
    const double tIni = MPI_Wtime();
    Image original;
    int dims[2] = {0, 0};
    if (rank == 0) {
        try {
            original = readPGM(inputPath.string());
            dims[0] = original.width;
            dims[1] = original.height;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    const double tLectura = MPI_Wtime() - tIni;

    // Todos necesitan conocer el tamano para reservar memoria.
    MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
    const int W = dims[0], H = dims[1];
    if (estrategia == "datos" && H < P) {
        if (rank == 0) std::cerr << "Error: hay mas procesos (" << P << ") que filas (" << H << ")\n";
        MPI_Finalize();
        return 1;
    }

    Resultado res;
    if (rank == 0) {
        res.edges.resize(static_cast<size_t>(W) * H);
        res.rgb.resize(static_cast<size_t>(W) * H * 3);
    }

    Tiempos t;
    double tParaleloLocal = 0;
    {
        EstrategiaDatos* datos = nullptr;
        EstrategiaTareas* tareas = nullptr;
        if (estrategia == "datos")
            datos = new EstrategiaDatos(W, H, rank, P);
        else
            tareas = new EstrategiaTareas(W, H, rank, P);

        // Todos empiezan a la vez a medir.
        MPI_Barrier(MPI_COMM_WORLD);
        const double t0 = MPI_Wtime();
        for (int rep = 0; rep < N; ++rep) {
            if (datos)
                datos->repeticion(original, res, t);
            else
                tareas->repeticion(original, res, t);
        }
        tParaleloLocal = MPI_Wtime() - t0;
        delete datos;
        delete tareas;
    }

    // El tiempo paralelo es el del proceso que MAS tarda (todos deben acabar).
    double tParalelo = 0, compMax = 0, compSum = 0, commMax = 0, commSum = 0;
    MPI_Reduce(&tParaleloLocal, &tParalelo, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t.computo, &compMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t.computo, &compSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t.comunicacion, &commMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t.comunicacion, &commSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const double densidadLineal =
            static_cast<double>(res.sospechosos) / static_cast<double>(W + H);
        const bool defecto = densidadLineal > UMBRAL_DENSIDAD;

        const double tEscIni = MPI_Wtime();
        const std::filesystem::path outDir =
            std::filesystem::path("salidas_mpi") / (inputPath.stem().string() + "_" + estrategia);
        std::filesystem::create_directories(outDir);
        const std::filesystem::path realzadaPath = outDir / "realzada.ppm";
        const std::filesystem::path bordesPath = outDir / "bordes.pgm";
        writePPM(realzadaPath.string(), W, H, res.rgb);
        writePGM(bordesPath.string(), W, H, res.edges);
        const double tEscritura = MPI_Wtime() - tEscIni;

        const double k = 1000.0;  // s -> ms
        const double tTotal = tLectura + tParalelo + tEscritura;
        // Con la estrategia de tareas solo trabajan min(P,4) procesos.
        const int utiles = estrategia == "tareas" ? std::min(P, 4) : P;

        std::cerr << "Imagen: " << inputPath.string() << " (" << W << "x" << H
                  << " px) | N=" << N << " | estrategia=" << estrategia << " | P=" << P
                  << " (utiles " << utiles << ")\n";
        std::cerr << "Densidad lineal: " << densidadLineal << " (umbral " << UMBRAL_DENSIDAD
                  << ") -> " << (defecto ? "DEFECTO DETECTADO" : "PIEZA OK") << "\n";
        std::cerr << "--- Tiempos en ms (" << N << " repeticion(es)) ---\n";
        std::cerr << "Lectura (P0): " << tLectura * k << "\n";
        std::cerr << "Parte paralela (max. procesos): " << tParalelo * k << "\n";
        std::cerr << "  computo       max " << compMax * k << " | medio " << compSum / utiles * k
                  << "\n";
        std::cerr << "  comunicacion  max " << commMax * k << " | medio " << commSum / utiles * k
                  << "\n";
        std::cerr << "Escritura (P0): " << tEscritura * k << "\n";
        std::cerr << "TIEMPO TOTAL: " << tTotal * k << "\n";
        std::cerr << "Salidas: " << realzadaPath.string() << ", " << bordesPath.string() << "\n";

        std::cout << estrategia << ',' << P << ',' << tLectura * k << ',' << tParalelo * k << ','
                  << compMax * k << ',' << compSum / utiles * k << ',' << commMax * k << ','
                  << commSum / utiles * k << ',' << tEscritura * k << ',' << tTotal * k << '\n';
    }

    MPI_Finalize();
    return 0;
}
