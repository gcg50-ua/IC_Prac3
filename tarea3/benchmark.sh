#!/usr/bin/env bash
# Un unico comando (portable Mac/Linux): compila defect_detection.cc con
# cada flag de la lista, lo ejecuta sobre las 5 imagenes reales del
# engranaje, y genera tres cosas, todas con la misma convencion de
# nombres <MAQUINA><fecha-hora>:
#
#   tiempos/tiempos<MAQUINA><fecha-hora>.csv    -> 25 filas de tiempos
#   remarks/remarks<MAQUINA><fecha-hora>.txt    -> avisos de vectorizacion
#                                                   de cada flag (Clang o
#                                                   GCC, detectado solo)
#   binarios_mac/ o binarios_linux/             -> los 5 ejecutables de
#                                                   la ULTIMA ejecucion
#                                                   (se sobrescriben cada
#                                                   vez, no se acumulan)
#
# tiempos/ y remarks/ SI acumulan un fichero nuevo por ejecucion (no se
# sobrescriben); binarios_mac/ y binarios_linux/ solo guardan la ultima.
#
# Cada ejecucion del binario repite el bloque de computo N veces sobre la
# misma imagen (fuerza bruta) para que el tiempo llegue al orden de
# segundos incluso en la combinacion mas rapida (imagen pequena + flag
# mas optimizado). Ver LEEME.txt para el razonamiento completo.
#
# Uso:            ./benchmark.sh
# Cambiar N:       N=100 ./benchmark.sh
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-60}"

MAQUINA="$(uname -s)"
case "$MAQUINA" in
    Darwin) MAQUINA="Mac" ;;
    Linux)  MAQUINA="Linux" ;;
esac
FECHA_HORA="$(date +"%Y%m%d-%H%M")"

mkdir -p tiempos remarks binarios_mac binarios_linux
CSV="tiempos/tiempos${MAQUINA}${FECHA_HORA}.csv"
REMARKS="remarks/remarks${MAQUINA}${FECHA_HORA}.txt"
if [ "$MAQUINA" = "Mac" ]; then
    DIR_BINARIOS="binarios_mac"
else
    DIR_BINARIOS="binarios_linux"
fi
rm -f "$DIR_BINARIOS"/*

# compile_flags.txt para el editor (clangd/IntelliSense): se regenera aqui
# en vez de guardarse como fichero suelto en el repo, y se queda en el
# directorio (no se borra al terminar) para que el editor lo siga usando.
cat > compile_flags.txt <<'EOF'
-std=c++17
-Wall
-Wextra
EOF

# Deteccion del compilador real detras de "g++": en Mac suele ser Clang
# (Apple lo llama "g++" pero no lo es), en Linux suele ser GCC de verdad.
# Cada uno usa flags distintos para pedir los avisos de vectorizacion.
if g++ --version 2>&1 | head -1 | grep -qi clang; then
    ES_CLANG=1
    REMARK_FLAGS="-Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize"
else
    ES_CLANG=0
    REMARK_FLAGS="-fopt-info-vec-optimized -fopt-info-vec-missed"
fi

IMAGENES=(
    entradas/engranaje320x320.pgm
    entradas/engranaje640x640.pgm
    entradas/engranaje1024x1024.pgm
    entradas/engranaje1280x1280.pgm
    entradas/engranaje2048x2048.pgm
)

nombres_flags=(sin O0 O1 O2 O3)
flags_valores=("" "-O0" "-O1" "-O2" "-O3")

for img in "${IMAGENES[@]}"; do
    if [ ! -f "$img" ]; then
        echo "Falta $img -- deberian estar las 5 imagenes engranajeTAMANOxTAMANO.pgm en entradas/"
        exit 1
    fi
done

echo "======================================================"
echo " Benchmark deteccion de defectos - $MAQUINA - $FECHA_HORA"
echo " Compilador: $(g++ --version | head -1) ($([ "$ES_CLANG" -eq 1 ] && echo Clang || echo GCC))"
echo " N (repeticiones por fuerza bruta): $N"
echo " Flags a probar: ${nombres_flags[*]}"
echo " Imagenes: ${#IMAGENES[@]}  ->  total de filas esperadas: $(( ${#IMAGENES[@]} * ${#nombres_flags[@]} ))"
echo " Salidas: $CSV"
echo "          $REMARKS"
echo "          $DIR_BINARIOS/ (5 binarios de esta ejecucion)"
echo "======================================================"

echo "flag,imagen,ancho_px,repeticiones_N,lectura_ms,blur_ms,sharpen_ms,sobelx_ms,sobely_ms,filtros_total_ms,suma_ms,pitagoras_ms,combinaciones_total_ms,marcado_ms,escritura_ms,tiempo_total_ms" > "$CSV"

{
    echo "Avisos de autovectorizacion - $MAQUINA - $FECHA_HORA"
    echo "Compilador: $(g++ --version | head -1)"
    echo "Flags de diagnostico usados: $REMARK_FLAGS"
    echo "================================================================"
} > "$REMARKS"

n_flags=${#nombres_flags[@]}
fila=0
total_filas=$(( ${#IMAGENES[@]} * n_flags ))
i=0
while [ "$i" -lt "$n_flags" ]; do
    nombre="${nombres_flags[$i]}"
    flag="${flags_valores[$i]}"
    bin="$DIR_BINARIOS/defect_detection_${nombre}"

    echo ""
    echo "--- [$((i + 1))/$n_flags] Compilando con: ${flag:-(sin flags)}  [etiqueta=$nombre] ---"
    inicio_compilacion=$(date +%s)
    {
        echo ""
        echo "=== FLAG: ${flag:-(sin flags)}  [etiqueta=$nombre] ==="
        # shellcheck disable=SC2086
        g++ $flag -std=c++17 $REMARK_FLAGS -o "$bin" defect_detection.cc
    } >> "$REMARKS" 2>&1
    echo "    compilado en $(( $(date +%s) - inicio_compilacion )) s -> $bin"

    for img in "${IMAGENES[@]}"; do
        fila=$((fila + 1))
        ancho=$(basename "$img" .pgm | sed -E 's/^engranaje([0-9]+)x[0-9]+$/\1/')
        printf "    [%2d/%2d] %-4s / %-24s (N=%s) ... " "$fila" "$total_filas" "$nombre" "$(basename "$img")" "$N"
        linea=$("$bin" "$img" "$N" 2>/dev/null | tail -1)
        total_fila="${linea##*,}"
        echo "${total_fila} ms"
        echo "$nombre,$(basename "$img"),$ancho,$N,$linea" >> "$CSV"
    done

    i=$((i + 1))
done

echo ""
echo "======================================================"
echo "Listo:"
echo "  $CSV ($total_filas filas)"
echo "  $REMARKS"
echo "  $DIR_BINARIOS/ ($(ls "$DIR_BINARIOS" | wc -l | tr -d ' ') binarios)"
echo "======================================================"
