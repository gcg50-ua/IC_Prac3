// defect_detection.cc
//
// Inspeccion visual automatica de piezas: banco de 4 filtros de convolucion
// (blur, sharpen, sobel-X, sobel-Y) sobre una imagen en escala de grises
// (formato PGM), combinados en dos imagenes de salida independientes:
//
//   - realzada.ppm : blur + sharpen, con las zonas sospechosas
//                    resaltadas en rojo -> para revision humana
//   - bordes.pgm   : sobel-X + sobel-Y -> mapa de bordes para
//                    decision automatica (umbral -> defecto SI/NO)
//
// Estructura de carpetas esperada (relativa al directorio de ejecucion):
//   entradas/<nombre>.pgm            -> imagen de entrada
//   salidas/<nombre>/realzada.ppm  y  salidas/<nombre>/bordes.pgm
//
// Parametro N (opcional, por defecto 1): numero de veces que se repite el
// procesado (4 filtros + 2 combinaciones + marcado) sobre la MISMA imagen
// ya cargada en memoria, antes de escribir las salidas una unica vez. Sirve
// para forzar por fuerza bruta que el tiempo total llegue al orden de
// segundos que pide la practica, sin cambiar el algoritmo ni el resultado
// (todas las repeticiones son deterministas y dan el mismo resultado). La
// lectura del fichero y la escritura de las salidas NO se repiten, solo el
// bloque de computo.
//
// Compilacion:  g++ -O2 -std=c++17 -o defect_detection defect_detection.cc
// Ejecucion:    ./defect_detection entradas/imagen1.pgm [N=1]
//
// La ULTIMA linea que imprime el programa es, sola, el tiempo total en
// milisegundos (todo lo demas va antes, para poder extraerla facilmente
// desde un script con `tail -1`).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

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

struct FloatImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
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
    file.get();  // consume el unico espacio en blanco tras la cabecera

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

void writePGM(const std::string& path, const Image& img) {
    std::ofstream file(path, std::ios::binary);
    file << "P5\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.pixels.data()),
               static_cast<std::streamsize>(img.pixels.size()));
}

struct ColorImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;  // RGB intercalado, 3 bytes por pixel
};

void writePPM(const std::string& path, const ColorImage& img) {
    std::ofstream file(path, std::ios::binary);
    file << "P6\n" << img.width << " " << img.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(img.pixels.data()),
               static_cast<std::streamsize>(img.pixels.size()));
}

// Aplica un kernel 3x3 a toda la imagen. Cada pixel de salida se calcula
// de forma independiente con la misma operacion (paralelismo de datos).
FloatImage convolve3x3(const Image& src, const float kernel[3][3]) {
    FloatImage out;
    out.width = src.width;
    out.height = src.height;
    out.pixels.resize(src.pixels.size());

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            float sum = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    sum += kernel[ky + 1][kx + 1] * src.at(x + kx, y + ky);
                }
            }
            out.pixels[static_cast<size_t>(y) * src.width + x] = sum;
        }
    }
    return out;
}

Image clampToImage(const FloatImage& f) {
    Image img;
    img.width = f.width;
    img.height = f.height;
    img.pixels.resize(f.pixels.size());
    for (size_t i = 0; i < f.pixels.size(); ++i) {
        img.pixels[i] = static_cast<uint8_t>(std::clamp(f.pixels[i], 0.0f, 255.0f));
    }
    return img;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " entradas/<nombre>.pgm [N=1]\n";
        return 1;
    }
    const std::filesystem::path inputPath = argv[1];
    const int N = argc > 2 ? std::max(1, std::atoi(argv[2])) : 1;

    try {
        const auto t0 = Clock::now();
        Image original = readPGM(inputPath.string());
        const auto tLecturaFin = Clock::now();
        const double tLectura = ms(t0, tLecturaFin);

        const std::filesystem::path outDir =
            std::filesystem::path("salidas") / inputPath.stem();
        std::filesystem::create_directories(outDir);

        static const float blurK[3][3] = {
            {1.f / 16, 2.f / 16, 1.f / 16},
            {2.f / 16, 4.f / 16, 2.f / 16},
            {1.f / 16, 2.f / 16, 1.f / 16}};
        static const float sharpenK[3][3] = {
            {0, -1, 0},
            {-1, 5, -1},
            {0, -1, 0}};
        static const float sobelXK[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}};
        static const float sobelYK[3][3] = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}};

        const int UMBRAL = 220;
        const double UMBRAL_DENSIDAD = 3.75;
        const float alpha = 0.65f;  // intensidad de la marca roja (0=nada, 1=rojo puro)

        // Acumuladores de tiempo: se suman las N repeticiones del bloque de
        // computo, para que la comparacion entre filtros/combinaciones siga
        // siendo valida (la proporcion entre ellos no cambia con N) a la vez
        // que el total se infla a proposito por fuerza bruta.
        double accBlur = 0, accSharpen = 0, accSobelX = 0, accSobelY = 0;
        double accSuma = 0, accPitagoras = 0, accMarcado = 0;

        // Resultados de la ultima repeticion (todas son identicas: mismo
        // calculo determinista sobre los mismos datos de entrada).
        Image enhanced, edges;
        ColorImage marcada;
        long pixelesSobreUmbral = 0;
        double densidadLineal = 0.0;
        bool defecto = false;

        const auto tComputoIni = Clock::now();
        for (int rep = 0; rep < N; ++rep) {
            const auto tBlurIni = Clock::now();
            Image blurred = clampToImage(convolve3x3(original, blurK));
            accBlur += ms(tBlurIni, Clock::now());

            const auto tSharpenIni = Clock::now();
            Image sharpened = clampToImage(convolve3x3(original, sharpenK));
            accSharpen += ms(tSharpenIni, Clock::now());

            const auto tSobelXIni = Clock::now();
            FloatImage gx = convolve3x3(original, sobelXK);
            accSobelX += ms(tSobelXIni, Clock::now());

            const auto tSobelYIni = Clock::now();
            FloatImage gy = convolve3x3(original, sobelYK);
            accSobelY += ms(tSobelYIni, Clock::now());

            // --- combinacion 1 (suma): blur + sharpen -> imagen realzada ---
            const auto tSumaIni = Clock::now();
            enhanced.width = original.width;
            enhanced.height = original.height;
            enhanced.pixels.resize(original.pixels.size());
            for (size_t i = 0; i < enhanced.pixels.size(); ++i) {
                int avg = (blurred.pixels[i] + sharpened.pixels[i]) / 2;
                enhanced.pixels[i] = static_cast<uint8_t>(std::clamp(avg, 0, 255));
            }
            accSuma += ms(tSumaIni, Clock::now());

            // --- combinacion 2 (pitagoras): sobel-X + sobel-Y -> mapa de bordes ---
            const auto tPitagorasIni = Clock::now();
            edges.width = original.width;
            edges.height = original.height;
            edges.pixels.resize(original.pixels.size());
            pixelesSobreUmbral = 0;
            std::vector<bool> sospechoso(original.pixels.size(), false);
            for (size_t i = 0; i < edges.pixels.size(); ++i) {
                const float mag = std::sqrt(gx.pixels[i] * gx.pixels[i] +
                                             gy.pixels[i] * gy.pixels[i]);
                const uint8_t value = static_cast<uint8_t>(std::clamp(mag, 0.0f, 255.0f));
                edges.pixels[i] = value;
                if (value > UMBRAL) {
                    sospechoso[i] = true;
                    ++pixelesSobreUmbral;
                }
            }
            accPitagoras += ms(tPitagorasIni, Clock::now());

            densidadLineal = static_cast<double>(pixelesSobreUmbral) /
                              static_cast<double>(original.width + original.height);
            defecto = densidadLineal > UMBRAL_DENSIDAD;

            // --- imagen para revision humana: realzada + zonas sospechosas en rojo ---
            const auto tMarcadoIni = Clock::now();
            marcada.width = original.width;
            marcada.height = original.height;
            marcada.pixels.resize(original.pixels.size() * 3);
            for (size_t i = 0; i < enhanced.pixels.size(); ++i) {
                const float gray = enhanced.pixels[i];
                uint8_t r, g, b;
                if (sospechoso[i]) {
                    r = static_cast<uint8_t>(gray * (1 - alpha) + 255 * alpha);
                    g = static_cast<uint8_t>(gray * (1 - alpha));
                    b = static_cast<uint8_t>(gray * (1 - alpha));
                } else {
                    r = g = b = static_cast<uint8_t>(gray);
                }
                marcada.pixels[i * 3 + 0] = r;
                marcada.pixels[i * 3 + 1] = g;
                marcada.pixels[i * 3 + 2] = b;
            }
            accMarcado += ms(tMarcadoIni, Clock::now());
        }
        const auto tComputoFin = Clock::now();

        const auto tEscrituraIni = Clock::now();
        const std::filesystem::path realzadaPath = outDir / "realzada.ppm";
        const std::filesystem::path bordesPath = outDir / "bordes.pgm";
        writePPM(realzadaPath.string(), marcada);
        writePGM(bordesPath.string(), edges);
        const auto tEscrituraFin = Clock::now();

        const auto t1 = Clock::now();

        const double tFiltrosTotal = accBlur + accSharpen + accSobelX + accSobelY;
        const double tCombinacionesTotal = accSuma + accPitagoras;
        const double tTotal = ms(t0, t1);

        const double tEscritura = ms(tEscrituraIni, tEscrituraFin);

        std::cerr << "Imagen: " << inputPath.string() << " (" << original.width << "x"
                  << original.height << " px) | N=" << N << "\n";
        std::cerr << "Densidad lineal: " << densidadLineal << " (umbral " << UMBRAL_DENSIDAD
                  << ") -> " << (defecto ? "DEFECTO DETECTADO" : "PIEZA OK") << "\n";
        std::cerr << "--- Tiempos acumulados en " << N << " repeticion(es), en ms ---\n";
        std::cerr << "Lectura (inicializacion): " << tLectura << "\n";
        std::cerr << "Filtro blur: " << accBlur << "\n";
        std::cerr << "Filtro sharpen: " << accSharpen << "\n";
        std::cerr << "Filtro sobel-X: " << accSobelX << "\n";
        std::cerr << "Filtro sobel-Y: " << accSobelY << "\n";
        std::cerr << "Total 4 filtros: " << tFiltrosTotal << "\n";
        std::cerr << "Combinacion suma: " << accSuma << "\n";
        std::cerr << "Combinacion pitagoras: " << accPitagoras << "\n";
        std::cerr << "Total 2 combinaciones (juntar los filtros): " << tCombinacionesTotal << "\n";
        std::cerr << "Generacion imagen marcada: " << accMarcado << "\n";
        std::cerr << "Escritura de salidas (codigo final, una vez): " << tEscritura << "\n";
        std::cerr << "Media por repeticion (computo/N): " << ms(tComputoIni, tComputoFin) / N
                   << "\n";
        std::cerr << "TIEMPO TOTAL: " << tTotal << "\n";
        std::cerr << "Salidas: " << realzadaPath.string() << ", " << bordesPath.string() << "\n";

        // Unica linea en stdout, formato CSV, para que benchmark.sh la capture
        // entera con `tail -1` y la vuelque como columnas del CSV final:
        //   lectura,blur,sharpen,sobelx,sobely,filtros_total,
        //   suma,pitagoras,combinaciones_total,marcado,escritura,total
        std::cout << tLectura << ',' << accBlur << ',' << accSharpen << ',' << accSobelX << ','
                   << accSobelY << ',' << tFiltrosTotal << ',' << accSuma << ',' << accPitagoras
                   << ',' << tCombinacionesTotal << ',' << accMarcado << ',' << tEscritura << ','
                   << tTotal << '\n';

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
