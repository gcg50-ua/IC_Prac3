#!/usr/bin/env bash
# Benchmark de la version MPI (tarea 3). Compila la version secuencial y la
# MPI con -O2, reparte el ejecutable MPI a los nodos del hostfile (si se da) y
# mide: 5 imagenes x {secuencial, datos, tareas} x numeros de procesos x RONDAS.
#
# Genera tiempos_mpi/tiempos_mpi<MAQUINA><fecha-hora>.csv con columnas:
#   estrategia,procesos,imagen,ancho_px,N,ronda,lectura_ms,paralelo_ms,
#   computo_max_ms,computo_medio_ms,comunicacion_max_ms,comunicacion_media_ms,
#   escritura_ms,total_ms
# (para "secuencial" paralelo_ms = computo, y comunicacion = 0)
#
# Uso (en la sala, en el equipo principal, tras haber hecho la tarea 2 para
# tener SSH sin contrasena):
#   cp -r tarea3 ~/tarea3 && cd ~/tarea3
#   HOSTFILE=../tarea2/hosts.txt ./benchmark_mpi.sh
#
# Variables opcionales:
#   N=60                    repeticiones del computo por ejecucion (como en P2)
#   NPS="1 2 3 4 6 8 12"    numeros de procesos a probar
#   RONDAS=3                ejecuciones de cada configuracion (se usa la mediana)
#   ESTRATEGIAS="datos tareas"
#   HOSTFILE=hosts.txt      sin el, todo se ejecuta en la maquina local
#   MPIRUN_EXTRA="--map-by node"  opciones extra para mpirun (Open MPI)
#   MPICXX=mpic++  MPIRUN=mpirun
set -euo pipefail
cd "$(dirname "$0")"

N="${N:-60}"
NPS="${NPS:-1 2 3 4 6 8 12}"
RONDAS="${RONDAS:-3}"
ESTRATEGIAS="${ESTRATEGIAS:-datos tareas}"
MPICXX="${MPICXX:-mpic++}"
MPIRUN="${MPIRUN:-mpirun}"
read -r -a EXTRA <<< "${MPIRUN_EXTRA:-}"
HOSTOPT=()
if [ -n "${HOSTFILE:-}" ]; then
    HOSTFILE="$(cd "$(dirname "$HOSTFILE")" && pwd)/$(basename "$HOSTFILE")"
    HOSTOPT=(-hostfile "$HOSTFILE")
fi

IMAGENES=(
    entradas/engranaje320x320.pgm
    entradas/engranaje640x640.pgm
    entradas/engranaje1024x1024.pgm
    entradas/engranaje1280x1280.pgm
    entradas/engranaje2048x2048.pgm
)

MAQUINA="$(uname -s)"
[ "$MAQUINA" = "Darwin" ] && MAQUINA="Mac"
FECHA_HORA="$(date +"%Y%m%d-%H%M")"
mkdir -p tiempos_mpi binarios_linux
CSV="tiempos_mpi/tiempos_mpi${MAQUINA}${FECHA_HORA}.csv"
SEQ=binarios_linux/defect_detection
BIN=binarios_linux/defect_detection_mpi

echo "== Compilando (-O2)"
g++ -O2 -std=c++17 -o "$SEQ" defect_detection.cc
"$MPICXX" -O2 -std=c++17 -o "$BIN" defect_detection_mpi.cc
chmod o+x "$BIN"

# El ejecutable debe estar en la MISMA ruta en todos los nodos. Las imagenes
# no hace falta copiarlas: solo el proceso 0 lee y escribe ficheros.
if [ -n "${HOSTFILE:-}" ]; then
    echo "== Copiando el ejecutable a los nodos de $HOSTFILE"
    MIS_IPS=" $(hostname -I 2>/dev/null || true) "
    while read -r ip _; do
        ip="${ip%%$'\r'}"
        [ -z "$ip" ] && continue
        [[ "$ip" == \#* ]] && continue
        [[ "$MIS_IPS" == *" $ip "* ]] && continue
        ssh -o BatchMode=yes "$ip" "mkdir -p '$PWD/binarios_linux'"
        scp -q "$BIN" "$ip:$PWD/$BIN"
        echo "   $ip OK"
    done < "$HOSTFILE"
fi

echo "estrategia,procesos,imagen,ancho_px,N,ronda,lectura_ms,paralelo_ms,computo_max_ms,computo_medio_ms,comunicacion_max_ms,comunicacion_media_ms,escritura_ms,total_ms" > "$CSV"
echo "== Midiendo (N=$N, procesos: $NPS, rondas: $RONDAS) -> $CSV"

for img in "${IMAGENES[@]}"; do
    nombre="$(basename "$img")"
    ancho=$(basename "$img" .pgm | sed -E 's/^engranaje([0-9]+)x[0-9]+$/\1/')
    for ronda in $(seq 1 "$RONDAS"); do
        # Secuencial: lectura,blur,sharpen,sobelx,sobely,filtros,suma,pit,comb,marcado,escritura,total
        linea=$("$SEQ" "$img" "$N" 2>/dev/null | tail -1)
        IFS=, read -r lec _ _ _ _ _ _ _ _ _ esc tot <<< "$linea"
        comp=$(awk -v t="$tot" -v l="$lec" -v e="$esc" 'BEGIN{print t-l-e}')
        echo "secuencial,1,$nombre,$ancho,$N,$ronda,$lec,$comp,$comp,$comp,0,0,$esc,$tot" >> "$CSV"
        printf "   %-26s secuencial    P=1  ronda %s: %s ms\n" "$nombre" "$ronda" "$tot"

        for est in $ESTRATEGIAS; do
            for np in $NPS; do
                linea=$("$MPIRUN" "${EXTRA[@]}" "${HOSTOPT[@]}" -n "$np" "$BIN" "$img" "$N" "$est" 2>/dev/null | tail -1)
                IFS=, read -r _ procs resto <<< "$linea"
                echo "$est,$procs,$nombre,$ancho,$N,$ronda,$resto" >> "$CSV"
                printf "   %-26s %-12s P=%-2s ronda %s: %s ms\n" "$nombre" "$est" "$np" "$ronda" "${linea##*,}"
            done
        done
    done
done

echo "== Listo: $CSV"
echo "   Graficas: python3 graficas_mpi.py $CSV"
