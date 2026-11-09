#!/usr/bin/env bash
# Comprueba que la version MPI produce EXACTAMENTE las mismas imagenes que la
# secuencial, para ambas estrategias y distintos numeros de procesos.
#
# Uso:  ./verificar_mpi.sh [imagen.pgm]          (por defecto la de 320x320)
#       NPS="1 2 3 4 5 8" ./verificar_mpi.sh
#       HOSTFILE=hosts.txt ./verificar_mpi.sh    (en el cluster de la sala)
#       MPIRUN_EXTRA="--map-by node" ./verificar_mpi.sh
#                                                (opciones extra para mpirun; --map-by node
#                                                 es solo para Open MPI. En la sala hay MPICH,
#                                                 que ya reparte por turnos: no hace falta)
set -euo pipefail
cd "$(dirname "$0")"

MPICXX="${MPICXX:-mpic++}"
MPIRUN="${MPIRUN:-mpirun}"
NPS="${NPS:-1 2 3 4 5 7 8}"
IMG="${1:-entradas/engranaje320x320.pgm}"
read -r -a EXTRA <<< "${MPIRUN_EXTRA:-}"
HOSTOPT=()
[ -n "${HOSTFILE:-}" ] && HOSTOPT=(-hostfile "$HOSTFILE")

mkdir -p binarios_linux
g++ -O2 -std=c++17 -o binarios_linux/defect_detection defect_detection.cc
"$MPICXX" -O2 -std=c++17 -o binarios_linux/defect_detection_mpi defect_detection_mpi.cc

# En el cluster el ejecutable tiene que estar en la misma ruta en cada nodo
if [ -n "${HOSTFILE:-}" ]; then
    MIS_IPS=" $(hostname -I 2>/dev/null || true) "
    while read -r ip _; do
        ip="${ip%%$'\r'}"
        [ -z "$ip" ] && continue
        [[ "$ip" == \#* ]] && continue
        [[ "$MIS_IPS" == *" $ip "* ]] && continue
        ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$ip" "mkdir -p '$PWD/binarios_linux'"
        scp -q -o ConnectTimeout=10 binarios_linux/defect_detection_mpi "$ip:$PWD/binarios_linux/"
        echo "Copiado a $ip"
    done < "$HOSTFILE"
fi

nombre=$(basename "$IMG" .pgm)
./binarios_linux/defect_detection "$IMG" 1 >/dev/null 2>&1
ref="salidas/$nombre"

fallos=0
for est in datos tareas; do
    for np in $NPS; do
        "$MPIRUN" "${EXTRA[@]}" "${HOSTOPT[@]}" -n "$np" ./binarios_linux/defect_detection_mpi "$IMG" 1 "$est" >/dev/null 2>&1
        out="salidas_mpi/${nombre}_${est}"
        if cmp -s "$ref/bordes.pgm" "$out/bordes.pgm" && cmp -s "$ref/realzada.ppm" "$out/realzada.ppm"; then
            echo "OK     $est  P=$np"
        else
            echo "FALLO  $est  P=$np"
            fallos=$((fallos + 1))
        fi
    done
done
[ "$fallos" -eq 0 ] && echo "Todas las salidas coinciden con la version secuencial." || exit 1
