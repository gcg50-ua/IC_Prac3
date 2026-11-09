#!/usr/bin/env bash
grep -q "$(printf '\r')" "$0" 2>/dev/null && sed -i 's/\r$//' "$0" && exec bash "$0" "$@"
# =============================================================================
# run_all.sh  -  Practica 3, Tarea 4 (kNN con MPI)   -   TODO EN UN SOLO SCRIPT
#
# Copia este fichero a la carpeta del proyecto del profesor (la que tiene knn.cc, knn.h,
# crossValidation.cc, mnist.cc, sample.cc, timer.h, get_dataset.sh ...) y ejecutalo:
#
#     chmod +x run_all.sh
#     ./run_all.sh                       # todo: compila, verifica, perfila (gprof), mide y dibuja
#     HOSTFILE=hosts.txt SIZES="4000 8000 20000" NPS="1 2 4 8 12 16" REPS=3 ./run_all.sh
#
# Funciona con MPICH (mpiexec/Hydra) y con Open MPI: detecta cual hay instalado y usa las opciones de
# mpirun y el formato de hostfile de cada uno (se puede forzar con MPI_IMPL=mpich o MPI_IMPL=ompi).
#
# Solo se prueban las versiones optimizadas (compiladas con -O2): el secuencial como referencia
# y las tres versiones MPI (paso 1: paralelismo de datos; paso 2: paralelismo funcional con
# ejecucion especulativa; paso 3: lo mismo con cola de tareas dinamica y ayuda de los evaluadores).
#
# El script lleva DENTRO los fuentes (al final del fichero): knn_seq.cpp, knn_mpi.cpp,
# knn_mpi_spec.cpp, graficas.py y los grafos de flujo (.dot). Los extrae en mpi_src/ (no toca
# los ficheros del profesor), y deja todo lo generado en build/ y resultados/ (OUT=...).
#
# Si el fichero viene de Windows (saltos de linea CRLF): 'bash run_all.sh' se arregla solo; si al
# ejecutar './run_all.sh' sale "bash\r: No existe el fichero", haz una vez:  sed -i 's/\r$//' run_all.sh
#
# ETAPAS (en este orden; sin argumentos se ejecutan las marcadas con *):
#   arch *       arquitectura.txt: procesador, nucleos, caches, memoria, red (de esta maquina y de cada nodo)
#   extract      extrae los fuentes embebidos a mpi_src/ (build ya lo hace; solo si quieres leerlos)
#   build *      comprueba herramientas y dataset y compila en build/
#   distribute * copia build/ al resto de nodos del HOSTFILE (ssh sin contrasena)
#   nodos        nodos.txt: donde se lanzan los procesos y si tienen binarios y dataset (verify ya la hace)
#   verify *     verificacion.txt: las versiones MPI dan EXACTAMENTE lo mismo que el secuencial?
#   gprof *      gprof/: perfil del secuencial y de cada version MPI (un informe por proceso + el total)
#   bench *      resultados.csv + logs/: tiempos de cada version, tamano, nº de procesos y repeticion
#   plots *      resumen.txt, speedup.png, eficiencia.png, ... (speed-up, eficiencia, Amdahl, ...)
#   grafos *     grafos/: grafos de flujo secuencial y paralelos en DOT (+ PNG/SVG si hay graphviz)
#   clean        borra lo que este script deja en build/ (binarios, gmon) y mpi_src/, y restos de
#                versiones antiguas del script. Con CLEAN_RESULTS=1 borra tambien la carpeta de resultados.
#   Ejemplos:  ./run_all.sh bench plots     ./run_all.sh build verify     ./run_all.sh clean
#
# VARIABLES (todas opcionales)
#   SIZES="2000 4000 8000"  tamanos N del ajuste. Para cada N: test = N/2 y entrenamiento final = 3N
#                           (mantiene la proporcion ~84 % ajuste / 16 % final del problema real, que
#                           es N=20000 -> 10000 test y 60000 entrenamiento). Con N=20000 el
#                           secuencial tarda ~20 min en un PC normal: usalo con REPS=1 o 2.
#   NPS="1 2 4 8"           numeros de procesos MPI. Por defecto: automatico segun los huecos
#                           (slots=) del HOSTFILE o los nucleos de esta maquina (1 2 3 4 6 8 12 ...).
#   REPS=1                  repeticiones de cada medida (recomendado 3 para los resultados finales)
#   HOSTFILE=hosts.txt      fichero de hosts (una linea por ordenador). Vale "nombre" a secas (como en el
#                           enunciado), "nombre slots=4" (Open MPI) o "nombre:4" (MPICH)
#   MPI_IMPL=               mpich | ompi (por defecto se detecta con 'mpirun --version')
#   MPI_EXTRA=""            opciones extra de mpirun (dependen de la implementacion; p. ej. en Open MPI
#                           "--map-by node --mca btl_tcp_if_include eth0")
#   FORCE=0                 con FORCE=1 las mediciones siguen aunque la verificacion falle
#   BENCH_VERSIONS="knn_mpi knn_mpi_spec knn_mpi_dyn"   que versiones medir (paso 1, paso 2, paso 3)
#   PROF_N=2000  PROF_NP=<min(4,slots)>   tamano N y numero de procesos con los que se pasa gprof
#   GPROF_LEVELS="O2"       niveles de optimizacion del perfilado ("O0 O2" para los dos);  GPROF_SEQ=1  perfila
#                           tambien el secuencial (0 = solo las versiones MPI)
#   VERIFY_N=1000  VERIFY_NPS="1 2 3 4"   tamano y procesos de la verificacion
#   MAXK=10 FOLDS=5         rango de K y numero de folds
#   OUT=resultados          carpeta de salida;   APPEND=1  anade a resultados.csv en vez de reescribirlo
#   RUN_TIMEOUT=0           segundos maximos por ejecucion (0 = sin limite; util para no quedarse
#                           colgado si un nodo falla)
#   BUILD=<dir>             carpeta de binarios y dataset (por defecto ./build, como en el proyecto)
# =============================================================================

set -u -o pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SELF")" && pwd)"
cd "$ROOT" || exit 1

SIZES=${SIZES:-"2000 4000 8000"}
NPS=${NPS:-}
REPS=${REPS:-1}
PROF_N=${PROF_N:-2000}
PROF_NP=${PROF_NP:-}
GPROF_LEVELS=${GPROF_LEVELS:-"O2"}
GPROF_SEQ=${GPROF_SEQ:-1}
VERIFY_N=${VERIFY_N:-1000}
VERIFY_NPS=${VERIFY_NPS:-"1 2 3 4"}
MAXK=${MAXK:-10}
FOLDS=${FOLDS:-5}
HOSTFILE=${HOSTFILE:-}
MPI_EXTRA=${MPI_EXTRA:-}
MPI_IMPL=${MPI_IMPL:-}
FORCE=${FORCE:-0}
BENCH_VERSIONS=${BENCH_VERSIONS:-"knn_mpi knn_mpi_spec knn_mpi_dyn"}
APPEND=${APPEND:-0}
OUT=${OUT:-resultados}
RUN_TIMEOUT=${RUN_TIMEOUT:-0}
CLEAN_RESULTS=${CLEAN_RESULTS:-0}

for _o in $GPROF_LEVELS; do
    [[ "$_o" == O0 || "$_o" == O2 ]] || { echo "[ERROR] GPROF_LEVELS solo admite O0 y/o O2 (has puesto '$_o')" >&2; exit 1; }
done
case "$OUT" in /*) OUT_DIR="$OUT" ;; *) OUT_DIR="$ROOT/$OUT" ;; esac
BUILD=${BUILD:-$ROOT/build}
case "$BUILD" in /*) ;; *) BUILD="$ROOT/$BUILD" ;; esac
MY="$ROOT/mpi_src"          # fuentes propios extraidos de este script
COMPILE_LOG="$OUT_DIR/compilacion.log"
if [[ -n "$HOSTFILE" ]]; then
    HOSTFILE="$(readlink -f "$HOSTFILE")"
    [[ -f "$HOSTFILE" ]] || { echo "[ERROR] No existe el HOSTFILE '$HOSTFILE'" >&2; exit 1; }
fi
# Dentro de contenedores se suele ser root: Open MPI lo rechaza salvo que se confirme.
if [[ "$(id -u)" == 0 ]]; then export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1; fi

# Implementacion de MPI: ompi (Open MPI) o mpich (MPICH/Hydra, tambien Intel MPI).
if [[ -z "$MPI_IMPL" ]]; then
    if command -v mpirun >/dev/null 2>&1 && ! mpirun --version 2>&1 | grep -qiE 'open ?mpi|open-rte|orte'; then
        MPI_IMPL=mpich
    else
        MPI_IMPL=ompi
    fi
fi
case "$MPI_IMPL" in mpich|ompi) ;; *) echo "[ERROR] MPI_IMPL debe ser mpich u ompi (has puesto '$MPI_IMPL')" >&2; exit 1 ;; esac

CSV_HEADER="version,rep,np,tuning,test,final_train,max_k,folds,best_k,test_acc,total_s,load_s,tune_s,final_s,comm_s,nT,nE,restarts,wasted_s"

log()  { printf '\n== %s ==\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '   [AVISO] %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

is_local() { [[ "$1" == localhost || "$1" == 127.0.0.1 || "$1" == "$(hostname)" || "${1%%.*}" == "$(hostname -s 2>/dev/null)" ]]; }

# Lee el fichero de hosts (formas admitidas: "nodo", "nodo slots=4", "nodo:4"; si un nodo sale en varias
# lineas se juntan). Imprime una linea por ordenador: "nodo huecos_indicados lineas_sin_huecos".
hosts_summary() {
    [[ -n "$HOSTFILE" ]] || return 0
    awk '
        !/^[[:space:]]*(#|$)/ {
            h = $1; n = -1
            if (h ~ /:[0-9]+$/) { n = substr(h, index(h, ":") + 1) + 0; h = substr(h, 1, index(h, ":") - 1) }
            for (i = 2; i <= NF; i++) {
                if ($i ~ /^slots=[0-9]+$/) n = substr($i, 7) + 0
                else if (i == 2 && $i ~ /^[0-9]+$/) n = $i + 0
            }
            if (!(h in seen)) { seen[h] = 1; ord[++k] = h }
            if (n >= 0) ex[h] += n; else pl[h]++
        }
        END { for (i = 1; i <= k; i++) print ord[i], ex[ord[i]] + 0, pl[ord[i]] + 0 }' "$HOSTFILE"
}
hosts_list() { hosts_summary | awk '{print $1}'; }

# Nucleos de un nodo (por ssh si no es este equipo); 1 si no se puede saber.
node_cores() {
    local h=$1 c
    if is_local "$h"; then c=$(nproc 2>/dev/null)
    else c=$(ssh -n -o BatchMode=yes -o ConnectTimeout=3 "$h" nproc 2>/dev/null); fi
    [[ "$c" =~ ^[0-9]+$ ]] && (( c >= 1 )) || c=1
    echo "$c"
}

# MPICH (Hydra) no entiende el hostfile de Open MPI ("nodo slots=4"): se le pasa uno temporal "nodo:huecos".
# Sin huecos indicados cada linea cuenta 1, asi los procesos se reparten entre todos los ordenadores.
HYDRA_HF=""
if [[ -n "$HOSTFILE" && "$MPI_IMPL" == mpich ]]; then
    HYDRA_HF="$(mktemp "${TMPDIR:-/tmp}/hosts_mpich.XXXXXX")"
    trap 'rm -f "$HYDRA_HF"' EXIT
    hosts_summary | awk '{ n = $2 + $3; if (n < 1) n = 1; print $1 ":" n }' > "$HYDRA_HF"
fi

# Tamanos derivados de N (con los topes del dataset MNIST: 60000 train, 10000 test).
test_of()  { local t=$(( $1 / 2 )); (( t > 10000 )) && t=10000; echo "$t"; }
final_of() { local f=$(( 3 * $1 )); (( f > 60000 )) && f=60000; echo "$f"; }

# Procesos que caben: huecos indicados en el HOSTFILE (si un nodo no los indica, sus nucleos) o nucleos de esta maquina.
slots_total() {
    if [[ -n "$HOSTFILE" ]]; then
        local h ex pl n s=0
        while read -r h ex pl; do
            if (( ex + pl == 0 )); then n=1
            elif (( ex == 0 && pl == 1 )); then n=$(node_cores "$h")
            else n=$(( ex + pl )); fi
            s=$(( s + n ))
        done < <(hosts_summary)
        echo "$s"
    else
        nproc 2>/dev/null || echo 4
    fi
}
default_nps() {
    local max c out=""
    max=$(slots_total); (( max >= 1 )) || max=1
    for c in 1 2 3 4 6 8 12 16 24 32 48 64; do (( c <= max )) && out+="$c "; done
    [[ " $out " == *" $max "* ]] || out+="$max"
    echo $out
}
[[ -n "$NPS" ]] || NPS=$(default_nps)
[[ -n "$PROF_NP" ]] || { PROF_NP=$(slots_total); (( PROF_NP > 4 )) && PROF_NP=4; (( PROF_NP < 2 )) && PROF_NP=2; }

with_timeout() { if (( RUN_TIMEOUT > 0 )); then timeout -k 10 "$RUN_TIMEOUT" "$@"; else "$@"; fi; }

# Ejecuta un programa SECUENCIAL desde build/ (rutas relativas train/ y test/) y quita
# los contadores de progreso "i/N" que imprime KNN::classifyAndEvaluate.
run_seq() {
    local prog=$1; shift
    (cd "$BUILD" && with_timeout "./$prog" "$@") 2>&1 | tr '\r' '\n' | grep -vE '^[0-9]+/[0-9]+$'
}

# Ejecuta un programa MPI desde build/. Con HOSTFILE usa los nodos del fichero.
# Open MPI: --oversubscribe y --hostfile.  MPICH: -f <hosts> (permite mas procesos que huecos sin avisar).
run_mpi() {
    local np=$1; shift
    local args=()
    if [[ "$MPI_IMPL" == ompi ]]; then
        args+=(--oversubscribe)
        [[ -n "$HOSTFILE" ]] && args+=(-mca plm_rsh_no_tree_spawn 1 --hostfile "$HOSTFILE")
    else
        [[ -n "$HOSTFILE" ]] && args+=(-f "$HYDRA_HF")
    fi
    # shellcheck disable=SC2086
    (cd "$BUILD" && with_timeout mpirun ${args[@]+"${args[@]}"} $MPI_EXTRA -np "$np" "$@")
}

mpi_version_line() {
    if [[ "$MPI_IMPL" == ompi ]]; then mpirun --version 2>&1 | head -1
    else echo "MPICH/Hydra $(mpirun --version 2>&1 | awk '/Version:/ {print $2; exit}')"; fi
}

# launch <version> <np> <N>: lanza una version con el tamano N (test y entrenamiento final derivados).
#   knn_seq | knn_mpi (paso 1) | knn_mpi_spec (paso 2, estatico) | knn_mpi_dyn (paso 3, dinamico)
launch() {
    local ver=$1 np=$2 N=$3 T F
    T=$(test_of "$N"); F=$(final_of "$N")
    case "$ver" in
        knn_seq)      run_seq knn_seq "$MAXK" "$FOLDS" "$N" "$T" "$F" ;;
        knn_mpi)      run_mpi "$np" ./knn_mpi "$MAXK" "$FOLDS" "$N" "$T" "$F" ;;
        knn_mpi_spec) run_mpi "$np" ./knn_mpi_spec "$MAXK" "$FOLDS" "$N" "$T" "$F" 0 1 0 ;;
        knn_mpi_dyn)  run_mpi "$np" ./knn_mpi_spec "$MAXK" "$FOLDS" "$N" "$T" "$F" 0 1 1 ;;
        *) die "Version desconocida: $ver" ;;
    esac
}

# Lineas que deben coincidir entre versiones: accuracy por K, mejor K y accuracy de test.
extract_res() {
    grep -E '^(K:|Best k|Test accuracy)' | sed -E 's/ completed in.*//; s/Best k: ([0-9]+).*/Best k: \1/'
}

need_binaries() {
    local b
    for b in knn_seq knn_mpi knn_mpi_spec; do
        [[ -x "$BUILD/$b" ]] || die "Falta build/$b: ejecuta primero './run_all.sh build'"
    done
}

# Escribe en $2 el fichero embebido $1 (secciones "#@@BEGIN nombre" ... "#@@END nombre" al final de este script).
emit() {
    mkdir -p "$(dirname "$2")"
    awk -v n="$1" '$0 == "#@@BEGIN " n {f=1; next} $0 == "#@@END " n {f=0} f' "$SELF" > "$2"
    [[ -s "$2" ]] || die "No se pudo extraer '$1' del script"
}

compile() {  # compile <comando...>: lo anota en compilacion.log y para si falla
    mkdir -p "$OUT_DIR"
    echo "\$ $*" >> "$COMPILE_LOG"
    "$@" >> "$COMPILE_LOG" 2>&1 || die "Error de compilacion: $* (ver $COMPILE_LOG)"
}

# Fuentes del profesor que usan nuestros programas.
check_project_files() {
    local t
    for t in knn.cc knn.h crossValidation.cc crossValidation.h mnist.cc mnist.h sample.cc sample.h timer.h; do
        [[ -f "$ROOT/$t" ]] || die "Falta $t en $ROOT: copia run_all.sh a la carpeta del proyecto del profesor"
    done
}
SEQ_SRC=("$MY/knn_seq.cpp" "$ROOT/crossValidation.cc" "$ROOT/knn.cc" "$ROOT/mnist.cc" "$ROOT/sample.cc")
MPI_SRC=("$MY/knn_mpi.cpp" "$ROOT/knn.cc" "$ROOT/mnist.cc" "$ROOT/sample.cc")
SPEC_SRC=("$MY/knn_mpi_spec.cpp" "$ROOT/knn.cc" "$ROOT/mnist.cc" "$ROOT/sample.cc")

ensure_dataset() {
    local need=(train/train-images-idx3-ubyte train/train-labels-idx1-ubyte test/t10k-images-idx3-ubyte test/t10k-labels-idx1-ubyte)
    local f missing=0
    for f in "${need[@]}"; do [[ -s "$BUILD/$f" ]] || missing=1; done
    if (( missing )); then
        info "Falta el dataset en build/: ejecutando get_dataset.sh (necesita wget y red)"
        if [[ -f "$ROOT/get_dataset.sh" ]]; then
            chmod +x "$ROOT/get_dataset.sh"
            (cd "$ROOT" && ./get_dataset.sh) > "$OUT_DIR/get_dataset.log" 2>&1 || warn "get_dataset.sh termino con error (ver get_dataset.log); se comprueba el dataset igualmente"
        fi
        missing=0
        for f in "${need[@]}"; do [[ -s "$BUILD/$f" ]] || missing=1; done
        (( missing )) && die "Sigue faltando el dataset. Copia los 4 archivos IDX a $BUILD/train y $BUILD/test."
    fi
    info "Dataset MNIST: OK"
}

# ------------------------------------------------------------------ arch ----
stage_arch() {
    log "Arquitectura del sistema"
    mkdir -p "$OUT_DIR"
    local f="$OUT_DIR/arquitectura.txt"
    {
        echo "Fecha: $(date)"
        echo "Equipo: $(hostname)"
        echo "--- uname -a";  uname -a
        echo "--- lscpu";     lscpu 2>/dev/null || echo "(lscpu no disponible)"
        echo "--- memoria";   free -h 2>/dev/null
        echo "--- caches (getconf)"; getconf -a 2>/dev/null | grep -i cache
        echo "--- red";       ip -br addr 2>/dev/null || echo "(ip no disponible)"
        echo "--- herramientas"
        g++ --version 2>/dev/null | head -1
        mpi_version_line
    } > "$f" 2>&1
    if [[ -n "$HOSTFILE" ]]; then
        local h
        for h in $(hosts_list); do
            {
                echo; echo "=================== NODO $h"
                ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" \
                    'hostname; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L1d|L1i|L2|L3"; free -h | head -2' 2>&1
            } >> "$f"
        done
    fi
    info "Guardado en $f"
}

# --------------------------------------------------------------- extract ----
stage_extract() {
    log "Extrayendo los fuentes embebidos a $MY"
    local f
    for f in knn_seq.cpp knn_mpi.cpp knn_mpi_spec.cpp graficas.py \
             grafos/secuencial.dot grafos/paso1_datos.dot grafos/paso2_3_especulativo.dot; do
        emit "$f" "$MY/$f"
    done
    info "Hecho (los ficheros del profesor no se tocan)"
}

# ----------------------------------------------------------------- build ----
stage_build() {
    log "Herramientas y dataset"
    local t
    for t in g++ mpic++ mpirun; do
        command -v "$t" >/dev/null 2>&1 || die "Falta '$t' (hace falta g++ y una implementacion de MPI: MPICH u Open MPI)"
        info "$t: $(command -v "$t")"
    done
    info "MPI: $(mpi_version_line)  [MPI_IMPL=$MPI_IMPL]"
    check_project_files
    mkdir -p "$BUILD" "$OUT_DIR"
    rm -rf "$MY"                                   # fuentes siempre frescos (y sin restos de versiones antiguas)
    rm -f "$BUILD"/knn_*_opt* "$BUILD"/knn_seq_pg_* "$BUILD"/knn_mpi_pg_* "$BUILD"/knn_mpi_spec_pg_* 2>/dev/null
    stage_extract
    ensure_dataset

    log "Compilacion con -O2 (salida en $COMPILE_LOG)"
    : > "$COMPILE_LOG"
    compile g++    -O2 -std=c++17 -I"$ROOT" "${SEQ_SRC[@]}"  -o "$BUILD/knn_seq"
    compile mpic++ -O2 -std=c++17 -I"$ROOT" "${MPI_SRC[@]}"   -o "$BUILD/knn_mpi"
    compile mpic++ -O2 -std=c++17 -I"$ROOT" "${SPEC_SRC[@]}"  -o "$BUILD/knn_mpi_spec"
    info "knn_seq, knn_mpi (paso 1), knn_mpi_spec (paso 2 = modo estatico, paso 3 = modo dinamico)"
}

# ------------------------------------------------------------ distribute ----
distribute_build() {   # copia build/ a cada nodo del HOSTFILE (misma ruta en todos)
    local h qdir
    qdir=$(printf '%q' "$BUILD")
    for h in $(hosts_list); do
        is_local "$h" && continue
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "mkdir -p $qdir"; then
            warn "No se pudo conectar con $h por ssh (hacen falta claves sin contrasena y el mismo usuario)"
            continue
        fi
        if command -v rsync >/dev/null 2>&1 && ssh -o BatchMode=yes "$h" 'command -v rsync' >/dev/null 2>&1; then
            rsync -a --exclude 'gmon*' "$BUILD/" "$h:$qdir/" && info "$h: copiado (rsync)" || warn "$h: fallo rsync"
        else
            # sin rsync: el dataset solo si falta, y siempre los binarios
            if ! ssh -o BatchMode=yes "$h" "[ -s $qdir/train/train-images-idx3-ubyte ]"; then
                scp -q -r "$BUILD/train" "$BUILD/test" "$h:$qdir/" || warn "$h: fallo scp del dataset"
            fi
            scp -q "$BUILD"/knn_* "$h:$qdir/" && info "$h: copiado (scp)" || warn "$h: fallo scp"
        fi
    done
}

stage_distribute() {
    [[ -n "$HOSTFILE" ]] || { log "Distribucion"; info "Sin HOSTFILE: nada que copiar"; return; }
    log "Copia de build/ al resto de nodos (misma ruta en todos)"
    distribute_build
    info "Recuerda: los binarios deben ser validos en todos los nodos (misma arquitectura y librerias MPI)."
    info "Si la carpeta del proyecto esta en un disco compartido (NFS), no hace falta copiar nada."
}

# ----------------------------------------------------------------- nodos ----
stage_nodos() {
    mkdir -p "$OUT_DIR"
    log "Nodos y reparto de procesos"
    local maxnp=1 np
    for np in $VERIFY_NPS $NPS; do (( np > maxnp )) && maxnp=$np; done
    {
        echo "# Un proceso por linea (np=$maxnp): nodo, directorio de trabajo, archivos de train/test y si existe el binario"
        run_mpi "$maxnp" sh -c 'echo "$(hostname) cwd=$(pwd) train=$(ls train 2>/dev/null | wc -l) test=$(ls test 2>/dev/null | wc -l) knn_mpi=$([ -x ./knn_mpi ] && echo si || echo NO)"' 2>&1 | sort
    } > "$OUT_DIR/nodos.txt"
    sed 's/^/   /' "$OUT_DIR/nodos.txt"
    if grep -qE 'NO$|train=0|test=0' "$OUT_DIR/nodos.txt"; then
        warn "Algun nodo no tiene los binarios o el dataset en build/. Ejecuta la etapa 'distribute'."
    fi
}

# ---------------------------------------------------------------- verify ----
stage_verify() {
    need_binaries
    mkdir -p "$OUT_DIR"
    stage_nodos

    local N=$VERIFY_N T F
    T=$(test_of "$N"); F=$(final_of "$N")
    log "Verificacion: N=$N (test=$T, entrenamiento final=$F), procesos: $VERIFY_NPS"
    local ref got raw ok=1 np ver res first_err="" first_tag=""
    ref=$(launch knn_seq 1 "$N" | extract_res)
    [[ -n "$ref" ]] || die "El secuencial no produjo resultados"
    {
        echo "Referencia: knn_seq (secuencial) con maxK=$MAXK folds=$FOLDS N=$N test=$T entrenamiento_final=$F"
        echo "$ref"
        echo
    } > "$OUT_DIR/verificacion.txt"

    for np in $VERIFY_NPS; do
        for ver in knn_mpi knn_mpi_spec knn_mpi_dyn; do
            [[ $ver != knn_mpi && $np -lt 2 ]] && continue   # los pasos 2 y 3 necesitan >= 2 procesos
            raw=$(launch "$ver" "$np" "$N" 2>&1 | tr '\r' '\n' | grep -vE '^[0-9]+/[0-9]+$')
            got=$(printf '%s\n' "$raw" | extract_res)
            if [[ "$got" == "$ref" ]]; then res="OK"; else
                res="FALLO"; ok=0
                {
                    echo "--- $ver np=$np: DIFERENCIAS respecto al secuencial"; diff <(echo "$ref") <(echo "$got")
                    echo "--- ultimas lineas de la salida:"; printf '%s\n' "$raw" | tail -6
                } >> "$OUT_DIR/verificacion.txt"
                if [[ -z "$first_err" ]]; then first_tag="$ver np=$np"; first_err=$(printf '%s\n' "$raw" | grep -v '^$' | head -6); fi
            fi
            printf '%-14s np=%-3s %s\n' "$ver" "$np" "$res" | tee -a "$OUT_DIR/verificacion.txt" | sed 's/^/   /'
        done
    done
    if (( ok )); then
        echo "RESULTADO: todas las versiones coinciden con el secuencial" | tee -a "$OUT_DIR/verificacion.txt" | sed 's/^/   /'
    else
        echo "RESULTADO: HAY DIFERENCIAS (ver verificacion.txt)" | tee -a "$OUT_DIR/verificacion.txt" | sed 's/^/   /'
        warn "Salida del primer fallo ($first_tag):"
        printf '%s\n' "$first_err" | sed 's/^/      /' >&2
        if [[ "$FORCE" != 1 ]]; then
            die "La verificacion ha fallado: se detiene aqui (FORCE=1 ./run_all.sh para seguir de todos modos)."
        fi
        warn "FORCE=1: se continua a pesar de la verificacion."
    fi
}

# ----------------------------------------------------------------- gprof ----
top_flat() {  # primeras filas del perfil plano de un informe gprof
    awk '/^Flat profile/{f=1;next} /^Call graph/{f=0} f && NF>=4 && $1 ~ /^[0-9]+(\.[0-9]+)?$/ {s=$0; if (length(s)>110) s=substr(s,1,107) "..."; print s; if (++n>=8) exit}' "$1"
}

cpu_of_flat() {  # segundos de CPU muestreados por gprof en un perfil plano (la ultima cifra acumulada)
    awk '/^Flat profile/{f=1;next} /^Call graph/{f=0} f && NF>=4 && $1 ~ /^[0-9]+(\.[0-9]+)?$/ { if ($2+0 > m) m=$2+0 } END { printf "%.2f", m }' "$1"
}

# Con -pg cada proceso MPI escribe gmon_rank<N>.<pid> en su directorio de trabajo (build/ de SU nodo).
gmon_clean() {   # borra los gmon_rank* de build/ en este equipo y en los nodos
    rm -f "$BUILD"/gmon_rank* 2>/dev/null
    [[ -n "$HOSTFILE" ]] || return 0
    local h qdir
    qdir=$(printf '%q' "$BUILD")
    for h in $(hosts_list); do
        is_local "$h" && continue
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "rm -f $qdir/gmon_rank*" >/dev/null 2>&1
    done
    return 0
}

gmon_collect() {  # gmon_collect <carpeta>: trae a <carpeta> los gmon_rank* de este equipo y de los nodos
    local dest=$1 h qdir
    qdir=$(printf '%q' "$BUILD")
    mkdir -p "$dest"
    mv "$BUILD"/gmon_rank* "$dest"/ 2>/dev/null
    [[ -n "$HOSTFILE" ]] || return 0
    for h in $(hosts_list); do
        is_local "$h" && continue
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "cd $qdir && tar cf - gmon_rank* 2>/dev/null && rm -f gmon_rank*" 2>/dev/null | tar xf - -C "$dest" 2>/dev/null
    done
    return 0
}

stage_gprof() {
    local t
    for t in g++ mpic++ mpirun gprof; do
        command -v "$t" >/dev/null 2>&1 || die "Falta '$t' (hace falta g++, gprof/binutils y una implementacion de MPI)"
    done
    check_project_files
    mkdir -p "$BUILD" "$OUT_DIR"
    [[ -f "$MY/knn_seq.cpp" ]] || stage_extract
    ensure_dataset
    local d="$OUT_DIR/gprof" O ver bin tag r f dir
    local N=$PROF_N np=$PROF_NP T F
    T=$(test_of "$N"); F=$(final_of "$N")
    rm -rf "$d"; mkdir -p "$d"
    local res="$d/resumen_gprof.txt"
    {
        echo "Resumen gprof (N=$N, test=$T, entrenamiento final=$F; versiones MPI con $np procesos)."
        echo "gprof mide TIEMPO DE CPU (muestreo cada 10 ms), no tiempo de pared: las esperas de MPI (bloqueos,"
        echo "usleep, barreras) no cuentan, asi que la CPU de un proceso puede ser mucho menor que el tiempo total."
        echo "Ese hueco es la espera o el desequilibrio del reparto. Cada proceso MPI genera su propio perfil."
        echo "Columnas del perfil: %tiempo  acumulado  propio  llamadas  ms/llamada  ms/llamada-total  funcion"
    } > "$res"

    log "Compilando con -pg (niveles: $GPROF_LEVELS)"
    for O in $GPROF_LEVELS; do
        compile g++    -$O -g -pg -std=c++17 -I"$ROOT" "${SEQ_SRC[@]}"  -o "$BUILD/knn_seq_pg_$O"
        compile mpic++ -$O -g -pg -std=c++17 -I"$ROOT" "${MPI_SRC[@]}"  -o "$BUILD/knn_mpi_pg_$O"
        compile mpic++ -$O -g -pg -std=c++17 -I"$ROOT" "${SPEC_SRC[@]}" -o "$BUILD/knn_mpi_spec_pg_$O"
    done
    [[ -n "$HOSTFILE" ]] && distribute_build

    if [[ "$GPROF_SEQ" == 1 ]]; then
        log "gprof sobre el secuencial (N=$N)"
        for O in $GPROF_LEVELS; do
            info "Ejecutando knn_seq_pg_$O ..."
            rm -f "$BUILD/gmon.out"
            run_seq "knn_seq_pg_$O" "$MAXK" "$FOLDS" "$N" "$T" "$F" > "$d/salida_seq_$O.txt"
            if [[ ! -f "$BUILD/gmon.out" ]]; then warn "No se genero gmon.out (secuencial, -$O)"; continue; fi
            mv "$BUILD/gmon.out" "$d/gmon_seq_$O.out"
            (cd "$BUILD" && gprof -b -p -q "./knn_seq_pg_$O" "$d/gmon_seq_$O.out") > "$d/perfil_seq_$O.txt" 2>&1
            {
                echo; echo "=== secuencial, -$O: $(grep -m1 'Total elapsed time' "$d/salida_seq_$O.txt" 2>/dev/null || echo 'sin tiempo total') ==="
                echo "Funciones con mas tiempo propio:"; top_flat "$d/perfil_seq_$O.txt"
            } >> "$res"
            info "perfil_seq_$O.txt generado"
        done
    fi

    log "gprof sobre las versiones MPI (N=$N, $np procesos)"
    local -a extra
    for ver in knn_mpi knn_mpi_spec knn_mpi_dyn; do
        for O in $GPROF_LEVELS; do
            case "$ver" in
                knn_mpi)      bin="knn_mpi_pg_$O";      extra=() ;;
                knn_mpi_spec) bin="knn_mpi_spec_pg_$O"; extra=(0 1 0) ;;
                knn_mpi_dyn)  bin="knn_mpi_spec_pg_$O"; extra=(0 1 1) ;;
            esac
            tag="${ver}_$O"
            dir="$d/gmon_$tag"
            info "Ejecutando $tag con $np procesos ..."
            gmon_clean
            run_mpi "$np" "./$bin" "$MAXK" "$FOLDS" "$N" "$T" "$F" ${extra[@]+"${extra[@]}"} > "$d/salida_$tag.txt" 2>&1
            gmon_collect "$dir"
            if ! ls "$dir"/gmon_rank* >/dev/null 2>&1; then warn "No se generaron gmon_rank* para $tag (ver salida_$tag.txt)"; continue; fi

            : > "$d/perfil_${tag}_procesos.txt"
            {
                echo; echo "=== $tag: $(grep -m1 'Total elapsed time' "$d/salida_$tag.txt" 2>/dev/null || echo 'sin tiempo total') ==="
                echo "CPU muestreada por proceso (rango: segundos):"
            } >> "$res"
            for r in $(seq 0 $(( np - 1 ))); do
                f=$(ls "$dir"/gmon_rank${r}.* 2>/dev/null | head -1)
                [[ -n "$f" ]] || continue
                gprof -b -p "$BUILD/$bin" "$f" > "$dir/.plano" 2>&1
                { echo "################ proceso (rango) $r"; cat "$dir/.plano"; echo; } >> "$d/perfil_${tag}_procesos.txt"
                printf '   rango %-3s %s s\n' "$r" "$(cpu_of_flat "$dir/.plano")" >> "$res"
            done
            rm -f "$dir/.plano"
            # perfil total: gprof suma los ficheros de todos los procesos si se le pasan varios
            # (se evita 'gprof -s', que con algunos binarios falla con "somebody miscounted")
            gprof -b -p -q "$BUILD/$bin" "$dir"/gmon_rank* > "$d/perfil_${tag}_total.txt" 2>&1
            if grep -q '^Flat profile' "$d/perfil_${tag}_total.txt"; then
                { echo "Funciones con mas tiempo propio (todos los procesos sumados):"; top_flat "$d/perfil_${tag}_total.txt"; } >> "$res"
            else
                warn "gprof no pudo sumar los perfiles de $tag (ver perfil_${tag}_total.txt)"
            fi
            info "$tag: perfil_${tag}_total.txt y perfil_${tag}_procesos.txt generados"
        done
    done
    sed 's/^/   /' "$res"
}

# ----------------------------------------------------------------- bench ----
run_one() {  # run_one <version> <np> <N> <rep>
    local ver=$1 np=$2 N=$3 rep=$4 logf line
    logf="$OUT_DIR/logs/${ver}_N${N}_np${np}_r${rep}.log"
    launch "$ver" "$np" "$N" > "$logf" 2>&1
    line=$(grep '^CSV,' "$logf" | tail -1)
    if [[ -z "$line" ]]; then
        warn "Fallo: $ver N=$N np=$np rep=$rep (ver $logf): $(tr '\r' '\n' < "$logf" | grep -vE '^[0-9]+/[0-9]+$' | grep -v '^$' | tail -1 | cut -c1-120)"
        return
    fi
    line=${line#CSV,}
    echo "${line%%,*},$rep,${line#*,}" >> "$OUT_DIR/resultados.csv"
    printf '   %-14s N=%-6s np=%-3s rep=%-2s total=%s s\n' "$ver" "$N" "$np" "$rep" "$(echo "$line" | cut -d, -f10)"
}

stage_bench() {
    need_binaries
    if [[ "$APPEND" != 1 || ! -f "$OUT_DIR/resultados.csv" ]]; then
        rm -rf "$OUT_DIR/logs"                      # sin restos de ejecuciones anteriores
        mkdir -p "$OUT_DIR/logs"
        echo "$CSV_HEADER" > "$OUT_DIR/resultados.csv"
    fi
    mkdir -p "$OUT_DIR/logs"
    log "Mediciones: SIZES='$SIZES' NPS='$NPS' REPS=$REPS ${HOSTFILE:+HOSTFILE=$HOSTFILE}"
    local N rep np ver
    for N in $SIZES; do
        info "--- N=$N (test=$(test_of "$N"), entrenamiento final=$(final_of "$N"))"
        for rep in $(seq 1 "$REPS"); do
            run_one knn_seq 1 "$N" "$rep"
            for np in $NPS; do
                for ver in $BENCH_VERSIONS; do
                    [[ $ver != knn_mpi && $np -lt 2 ]] && continue   # los pasos 2 y 3 necesitan >= 2 procesos
                    run_one "$ver" "$np" "$N" "$rep"
                done
            done
        done
    done
    info "Tiempos en $OUT_DIR/resultados.csv, salidas completas en $OUT_DIR/logs/"
}

# ----------------------------------------------------------------- plots ----
stage_plots() {
    log "Resumen y graficas"
    [[ -s "$OUT_DIR/resultados.csv" ]] || die "No hay $OUT_DIR/resultados.csv: ejecuta antes la etapa 'bench'"
    command -v python3 >/dev/null 2>&1 || { warn "No hay python3: solo queda el CSV"; return; }
    [[ -f "$MY/graficas.py" ]] || emit graficas.py "$MY/graficas.py"
    python3 "$MY/graficas.py" "$OUT_DIR/resultados.csv" "$OUT_DIR" || warn "graficas.py termino con error"
}

# ---------------------------------------------------------------- grafos ----
stage_grafos() {
    log "Grafos de flujo (DOT)"
    local d="$OUT_DIR/grafos" f
    rm -rf "$d"; mkdir -p "$d"
    for f in secuencial paso1_datos paso2_3_especulativo; do
        emit "grafos/$f.dot" "$d/$f.dot"
        if command -v dot >/dev/null 2>&1; then
            dot -Tpng "$d/$f.dot" -o "$d/$f.png" && dot -Tsvg "$d/$f.dot" -o "$d/$f.svg" && info "$f: .dot, .png y .svg"
        else
            info "$f: solo .dot (instala graphviz: sudo apt install graphviz, o pega el .dot en https://dreampuf.github.io/GraphvizOnline)"
        fi
    done
}

# ----------------------------------------------------------------- clean ----
stage_clean() {
    log "Limpieza"
    rm -f "$BUILD"/knn_seq* "$BUILD"/knn_mpi* "$BUILD"/gmon* 2>/dev/null
    rm -rf "$MY"
    info "Borrados los binarios de $BUILD (el dataset se conserva) y $MY"
    if [[ -n "$HOSTFILE" ]]; then
        local h qdir
        qdir=$(printf '%q' "$BUILD")
        for h in $(hosts_list); do
            is_local "$h" && continue
            ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" "rm -f $qdir/knn_seq* $qdir/knn_mpi* $qdir/gmon*" >/dev/null 2>&1 && info "$h: binarios borrados"
        done
    fi
    if [[ "$CLEAN_RESULTS" == 1 ]]; then
        rm -rf "$OUT_DIR"
        info "Borrada la carpeta de resultados $OUT_DIR"
    else
        # restos de versiones antiguas del script dentro de los resultados
        rm -rf "$OUT_DIR/tarea0" 2>/dev/null
        info "Resultados conservados en $OUT_DIR (CLEAN_RESULTS=1 ./run_all.sh clean para borrarlos tambien)"
    fi
}

# ----------------------------------------------------------------- leeme ----
write_leeme() {
    cat > "$OUT_DIR/LEEME.txt" <<'EOF'
Contenido de la carpeta de resultados (lo genera run_all.sh)
  arquitectura.txt     procesador, nucleos, caches, memoria y red (de esta maquina y de cada nodo)  -> memoria
  compilacion.log      ordenes de compilacion y avisos
  nodos.txt            en que nodos arrancan los procesos y si tienen binarios y dataset
  verificacion.txt     las versiones MPI dan exactamente lo mismo que el secuencial?
  gprof/               perfiles gprof (-O2) del secuencial y de cada version MPI:
                         perfil_seq_<O>.txt                  secuencial: perfil plano + grafo de llamadas
                         perfil_<version>_<O>_total.txt      todos los procesos sumados (perfil plano + grafo de llamadas)
                         perfil_<version>_<O>_procesos.txt   perfil plano de cada proceso (rango) por separado
                         resumen_gprof.txt                   CPU de cada proceso y funciones mas costosas
                         salida_*.txt, gmon_*                salida del programa y datos en bruto de gprof
  resultados.csv       una fila por ejecucion (version, rep, np, tamanos, K, accuracy, tiempos, ...)
  logs/                la salida completa de cada ejecucion
  resumen.txt          speed-up, eficiencia, Karp-Flatt, comunicacion, cota de Amdahl, consistencia
  speedup.png eficiencia.png tiempos_knn_mpi.png tiempos_versiones.png
  grafos/              grafos de flujo: secuencial, paso 1, pasos 2-3 (.dot, .png, .svg)
Versiones (columna "version" del CSV):
  knn_seq        secuencial (referencia)              knn_mpi_spec   paso 2: funcional + especulativa (estatica)
  knn_mpi        paso 1: paralelismo de datos         knn_mpi_dyn    paso 3: cola dinamica + ayuda de evaluadores
EOF
}

# ------------------------------------------------------------------ main ----
ALL_STAGES="arch extract build distribute nodos verify gprof bench plots grafos clean"
DEFAULT_STAGES="arch build distribute verify gprof bench plots grafos"
REQUESTED=()
for a in "$@"; do a=${a//$'\r'/}; [[ -n "$a" ]] && REQUESTED+=("$a"); done
(( ${#REQUESTED[@]} )) || REQUESTED=(default)

for s in "${REQUESTED[@]}"; do
    [[ "$s" == default ]] && continue
    [[ " $ALL_STAGES " == *" $s "* ]] || die "Etapa desconocida '$s'. Etapas: $ALL_STAGES"
done

wanted() {
    local r
    for r in "${REQUESTED[@]}"; do
        [[ "$r" == "$1" ]] && return 0
        [[ "$r" == default && " $DEFAULT_STAGES " == *" $1 "* ]] && return 0
    done
    return 1
}

if wanted clean; then
    stage_clean
    exit 0
fi

mkdir -p "$OUT_DIR"
T_START=$SECONDS
for s in $ALL_STAGES; do
    [[ "$s" == clean ]] && continue
    # 'nodos' ya forma parte de 'verify' y 'extract' de 'build': solo si se piden a mano y sin ellos
    [[ "$s" == nodos ]] && wanted verify && continue
    [[ "$s" == extract ]] && wanted build && continue
    wanted "$s" && "stage_$s"
done
write_leeme

log "Terminado en $(( SECONDS - T_START )) s. Resultados en $OUT_DIR"
ls -1 "$OUT_DIR" | sed 's/^/   /'
exit 0

# ===================== FICHEROS EMBEBIDOS (no editar a mano; los lee 'emit') =====================
: <<'__PAYLOAD_NO_EJECUTAR__'
#@@BEGIN knn_seq.cpp
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
#@@END knn_seq.cpp
#@@BEGIN knn_mpi.cpp
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
#@@END knn_mpi.cpp
#@@BEGIN knn_mpi_spec.cpp
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
#@@END knn_mpi_spec.cpp
#@@BEGIN graficas.py
#!/usr/bin/env python3
"""Resumen y graficas de resultados.csv (lo lanza run_all.sh; tambien se puede usar a mano).

    python3 graficas.py resultados/resultados.csv resultados

Genera:
  resumen.txt            tiempos, speed-up, eficiencia, fraccion serie (Karp-Flatt), peso de la
                         comunicacion, cota de Amdahl y comprobacion de que todas las versiones
                         dan el mismo resultado
  speedup.png            speed-up frente a procesos (una grafica por tamano) y la recta ideal
  eficiencia.png         eficiencia = speed-up / procesos
  tiempos_knn_mpi.png    reparto del tiempo del paso 1 (carga, ajuste, evaluacion, comunicacion)
  tiempos_versiones.png  tiempo total de cada version (tamano mayor)
Sin matplotlib solo se genera resumen.txt.

Versiones (columna "version" del CSV):
  knn_seq        secuencial (referencia)          knn_mpi_spec   paso 2: funcional + especulativa
  knn_mpi        paso 1: paralelismo de datos     knn_mpi_dyn    paso 3: cola dinamica + ayuda

Definiciones (T = tiempo total medio de las repeticiones):
  speed-up(N, p)  = T_secuencial(N) / T_paralelo(N, p)   (mismo N y mismos argumentos)
  eficiencia      = speed-up / p
  Karp-Flatt      e = (1/S - 1/p) / (1 - 1/p): fraccion serie "efectiva" medida. Si crece con p hay
                  sobrecoste de comunicacion, sincronizacion o desequilibrio; si es constante es
                  la parte realmente secuencial (ley de Amdahl).
  Amdahl          la carga del dataset la repite cada proceso, asi que no se acelera. Con
                  f = carga / T_secuencial, el speed-up maximo teorico es 1 / f. Es una cota
                  MINIMA de la fraccion secuencial (hay mas partes no paralelizables).
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

if len(sys.argv) < 3:
    sys.exit("Uso: graficas.py resultados.csv carpeta_salida")
csv_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

NAMES = {
    "knn_mpi": "paso 1: datos",
    "knn_mpi_spec": "paso 2: especulativa",
    "knn_mpi_dyn": "paso 3: dinamica+ayuda",
}
COLORS = {"knn_mpi": "#1f77b4", "knn_mpi_spec": "#d62728", "knn_mpi_dyn": "#2ca02c"}
MPI = ["knn_mpi", "knn_mpi_spec", "knn_mpi_dyn"]


def num(x):
    try:
        return float(x) if x not in ("", None) else None
    except ValueError:
        return None


rows = list(csv.DictReader(open(csv_path, newline="")))
if not rows:
    sys.exit("resultados.csv esta vacio")

groups = defaultdict(list)  # (version, N, np) -> filas
for r in rows:
    groups[(r["version"], int(r["tuning"]), int(r["np"]))].append(r)


def mean(key, col):
    vals = [num(r[col]) for r in groups[key] if num(r[col]) is not None]
    return st.mean(vals) if vals else None


sizes = sorted({k[1] for k in groups})
versions = [v for v in MPI if any(k[0] == v for k in groups)]


def nps_of(v, n):
    return sorted(k[2] for k in groups if k[0] == v and k[1] == n and mean(k, "total_s"))


tseq = {n: mean(("knn_seq", n, 1), "total_s") for n in sizes if ("knn_seq", n, 1) in groups}

# ------------------------------------------------------------------ resumen
L = []
L.append("RESUMEN  (tiempos en segundos; media de las repeticiones)")
L.append("speed-up = T_secuencial / T_paralelo (mismo N);  eficiencia = speed-up / procesos;  KF = Karp-Flatt (fraccion serie efectiva)")
L.append("")
for n in sizes:
    any_row = next(r for k, rs in groups.items() if k[1] == n for r in rs)
    L.append(f"=== N = {n}  (test = {any_row['test']}, entrenamiento final = {any_row['final_train']})")
    t0 = tseq.get(n)
    if t0 is None:
        L.append("   (sin medida secuencial: no se pueden calcular speed-up ni eficiencia)")
    else:
        load = mean(("knn_seq", n, 1), "load_s") or 0.0
        tune = mean(("knn_seq", n, 1), "tune_s") or 0.0
        fin = mean(("knn_seq", n, 1), "final_s") or 0.0
        L.append(f"   secuencial: total {t0:.3f}  (carga {load:.3f}, ajuste {tune:.3f} = {100 * tune / t0:.0f} %, "
                 f"evaluacion final {fin:.3f} = {100 * fin / t0:.0f} %)")
        if load > 0:
            f = load / t0
            L.append(f"   Amdahl: fraccion secuencial medida (carga replicada) f = {f:.4f}  ->  speed-up maximo teorico <= {1 / f:.1f}")
    L.append(f"   {'version':<24}{'p':>4}{'total':>10}{'speed-up':>10}{'efic.':>7}{'KF':>8}   notas")
    for v in versions:
        for p in nps_of(v, n):
            t = mean((v, n, p), "total_s")
            s = (t0 / t) if t0 else None
            e = (s / p) if s else None
            kf = ((1 / s - 1 / p) / (1 - 1 / p)) if (s and p > 1) else None
            note = ""
            if v == "knn_mpi":
                comm = mean((v, n, p), "comm_s")
                if comm is not None:
                    note = f"comunicacion+espera={comm:.3f}s ({100 * comm / t:.1f} % del total)"
            else:
                nt, ne = mean((v, n, p), "nT"), mean((v, n, p), "nE")
                rs, ws = mean((v, n, p), "restarts"), mean((v, n, p), "wasted_s")
                if None not in (nt, ne, rs, ws):
                    note = f"ajuste={int(nt)} eval={int(ne)} reinicios={rs:.1f} descartado={ws:.2f}s"
            f2 = lambda x, d=2: (f"{x:.{d}f}" if x is not None else "-")  # noqa: E731
            L.append(f"   {NAMES[v]:<24}{p:>4}{t:>10.3f}{f2(s):>10}{f2(e):>7}{f2(kf, 3):>8}   {note}")
    L.append("")

L.append("COMPROBACION DE CONSISTENCIA (mejor K y accuracy de test, por tamano; deben ser iguales en TODAS las versiones)")
for n in sizes:
    combos = {(r["best_k"], r["test_acc"]) for k, rs in groups.items() if k[1] == n for r in rs}
    state = "OK" if len(combos) == 1 else "AVISO: resultados distintos entre ejecuciones -> " + str(sorted(combos))
    L.append(f"   N={n}: {state}  {sorted(combos)[0] if len(combos) == 1 else ''}")
text = "\n".join(L) + "\n"
with open(os.path.join(out_dir, "resumen.txt"), "w", encoding="utf-8") as fh:
    fh.write(text)
print(text)

# ------------------------------------------------------------------ graficas
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # noqa: BLE001
    print(f"(matplotlib no disponible: {exc}; solo se ha generado resumen.txt)")
    sys.exit(0)


def panel_figure(metric, ylabel, fname, ideal):
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.8 * len(sizes), 4.1), squeeze=False)
    for ax, n in zip(axes[0], sizes):
        t0 = tseq.get(n)
        if t0 is None:
            ax.set_title(f"N={n} (sin secuencial)")
            continue
        allp = set()
        for v in versions:
            ps = nps_of(v, n)
            if ps:
                ax.plot(ps, [metric(t0 / mean((v, n, p), "total_s"), p) for p in ps], marker="o",
                        label=NAMES[v], color=COLORS[v])
                allp.update(ps)
        if allp:
            ps = sorted(allp)
            ax.plot(ps, [ideal(p) for p in ps], "k:", linewidth=1, label="ideal")
            ax.set_xticks(ps)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("procesos MPI")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=130)
    plt.close(fig)


if tseq and versions:
    panel_figure(lambda s, p: s, "speed-up", "speedup.png", lambda p: p)
    panel_figure(lambda s, p: s / p, "eficiencia", "eficiencia.png", lambda p: 1.0)

# reparto de tiempo del paso 1 (tamano mayor)
if "knn_mpi" in versions:
    n = max(n for n in sizes if nps_of("knn_mpi", n))
    ps = nps_of("knn_mpi", n)
    parts = [
        ("carga (replicada)", "load_s", "#9aa5b1"),
        ("ajuste: calculo", "tune_s", "#1f77b4"),
        ("evaluacion final: calculo", "final_s", "#2ca02c"),
        ("comunicacion + espera", "comm_s", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bottom = [0.0] * len(ps)
    for lab, col, color in parts:
        vals = [mean(("knn_mpi", n, p), col) or 0.0 for p in ps]
        ax.bar([str(p) for p in ps], vals, bottom=bottom, label=lab, color=color)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title(f"Paso 1: reparto del tiempo (N = {n}); cada fase = maximo entre procesos")
    ax.title.set_fontsize(9)
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("segundos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempos_knn_mpi.png"), dpi=130)
    plt.close(fig)

# tiempo total de cada version (tamano mayor)
if versions:
    n = max(sizes)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for v in versions:
        ps = nps_of(v, n)
        if ps:
            ax.plot(ps, [mean((v, n, p), "total_s") for p in ps], marker="o", label=NAMES[v], color=COLORS[v])
    if tseq.get(n):
        ax.axhline(tseq[n], color="gray", linestyle=":", linewidth=1, label="secuencial")
    ax.set_title(f"Tiempo total (N = {n})")
    ax.set_xlabel("procesos MPI")
    ax.set_ylabel("segundos")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tiempos_versiones.png"), dpi=130)
    plt.close(fig)

print("Graficas guardadas en", out_dir)
#@@END graficas.py
#@@BEGIN grafos/secuencial.dot
// Grafo de control de flujo de la solucion SECUENCIAL (main.cc / knn.cc / crossValidation.cc)
digraph secuencial {
  rankdir=TB; nodesep=0.35; ranksep=0.4;
  node [shape=box, fontname="Helvetica", fontsize=10, style=rounded];
  edge [fontname="Helvetica", fontsize=9];

  start  [shape=oval, label="main"];
  load   [label="loadDataset\n60000 entrenamiento + 10000 test\n(28x28 -> 14x14, normalizadas)"];
  subset [label="subconjunto de ajuste\n(primeras 20000 muestras)"];
  start -> load -> subset;

  subgraph cluster_tuning {
    label="fineTuning: K = 1..10, 5 folds   (~84 % del tiempo)"; style=dashed; fontname="Helvetica"; fontsize=11;
    kloop    [shape=diamond, style="", label="para cada K\n(1..10)"];
    foldloop [shape=diamond, style="", label="para cada fold\n(0..4)"];
    split    [label="splitDataset\nentrenamiento 16000 / validacion 4000"];
    build    [label="KNN(entrenamiento, K)\n(copia las 16000 muestras)"];
    sloop    [shape=diamond, style="", label="para cada muestra\nde validacion (4000)"];
    classify [label="classify(muestra)\n1) distancia euclidea a las 16000\n2) std::sort de las 16000\n3) votos de los K primeros -> moda"];
    acc      [label="accuracy del fold"];
    mean     [label="accuracy(K) = media de los 5 folds"];
    best     [label="si accuracy(K) > mejor:\nmejor K = K"];
    kloop -> foldloop -> split -> build -> sloop;
    sloop -> classify [label="siguiente"];
    classify -> sloop;
    sloop -> acc [label="fin"];
    acc -> foldloop [label="siguiente fold"];
    foldloop -> mean [label="fin de folds"];
    mean -> best;
    best -> kloop [label="siguiente K"];
  }
  subset -> kloop;

  final [label="evaluacion final   (~16 % del tiempo)\nKNN(60000 entrenamiento, mejor K)\nclassify de las 10000 muestras de test"];
  best -> final [label="fin de K\nDEPENDENCIA: necesita el mejor K", color=red, fontcolor=red];
  out [shape=oval, label="Best k, Test accuracy"];
  final -> out;

  n1 [shape=note, style="", fontsize=9, label="INDEPENDIENTES:\n- las 50 tareas (K, fold) entre si\n- las muestras de una validacion entre si\n- las muestras de test entre si"];
  n2 [shape=note, style="", fontsize=9, label="DEPENDENCIAS:\n- media de folds necesita los 5 folds\n- mejor K necesita todos los K\n- evaluacion final necesita el mejor K\n  (se rompe con ejecucion especulativa)"];
  n1 -> classify [style=dotted, arrowhead=none];
  n2 -> final [style=dotted, arrowhead=none];
}
#@@END grafos/secuencial.dot
#@@BEGIN grafos/paso1_datos.dot
// Solucion paralela PASO 1: paralelismo de datos (SPMD, memoria distribuida, paso de mensajes)
digraph paso1 {
  rankdir=TB; nodesep=0.4; ranksep=0.45; compound=true;
  node [shape=box, fontname="Helvetica", fontsize=10, style=rounded];
  edge [fontname="Helvetica", fontsize=9];

  init [shape=oval, label="MPI_Init / MPI_Comm_rank / MPI_Comm_size"];

  subgraph cluster_p0 { label="proceso 0"; style=dashed; fontname="Helvetica";
    l0 [label="carga del dataset\n(replicada)"]; t0 [label="ajuste: tareas t = 0, P, 2P, ...\n(reparto ciclico)"];
    k0 [label="elige mejor K\n(mismo en todos)"]; e0 [label="evaluacion final:\nbloque 0 del test"];
    l0 -> t0; t0 -> k0 [style=invis]; k0 -> e0;
  }
  subgraph cluster_p1 { label="proceso 1"; style=dashed; fontname="Helvetica";
    l1 [label="carga del dataset\n(replicada)"]; t1 [label="ajuste: tareas t = 1, P+1, ...\n(reparto ciclico)"];
    k1 [label="elige mejor K\n(mismo en todos)"]; e1 [label="evaluacion final:\nbloque 1 del test"];
    l1 -> t1; t1 -> k1 [style=invis]; k1 -> e1;
  }
  subgraph cluster_pn { label="proceso P-1"; style=dashed; fontname="Helvetica";
    ln [label="carga del dataset\n(replicada)"]; tn [label="ajuste: tareas t = P-1, 2P-1, ...\n(reparto ciclico)"];
    kn [label="elige mejor K\n(mismo en todos)"]; en [label="evaluacion final:\nbloque P-1 del test"];
    ln -> tn; tn -> kn [style=invis]; kn -> en;
  }

  init -> l0; init -> l1; init -> ln;

  allr [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Allreduce(SUM) de las 50 accuracies\nSINCRONIZACION: todos esperan al mas lento\n(cada proceso aporta solo sus posiciones; el resto 0.0)"];
  t0 -> allr; t1 -> allr; tn -> allr;
  allr -> k0; allr -> k1; allr -> kn;

  red [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Reduce(SUM) de los aciertos -> proceso 0"];
  e0 -> red; e1 -> red; en -> red;
  out [shape=oval, label="proceso 0: Best k, Test accuracy\nMPI_Reduce(MAX) de tiempos + MPI_Gather de metricas"];
  red -> out;
}
#@@END grafos/paso1_datos.dot
#@@BEGIN grafos/paso2_3_especulativo.dot
// Solucion paralela PASOS 2 y 3: paralelismo funcional + ejecucion especulativa
digraph especulativo {
  rankdir=TB; nodesep=0.5; ranksep=0.6; compound=true;
  node [shape=box, fontname="Helvetica", fontsize=10, style=rounded];
  edge [fontname="Helvetica", fontsize=9];

  subgraph cluster_T { label="GRUPO DE AJUSTE (procesos 0 .. nT-1)"; style=dashed; fontname="Helvetica"; fontsize=11;
    ta [label="proceso de ajuste\nestatico: tareas (k, fold) ciclicas\ndinamico: pide una tarea, la calcula,\nenvia el resultado, pide otra"];
  }

  subgraph cluster_E { label="GRUPO DE EVALUACION (procesos nT .. P-1; lider = nT)  -  comunicador propio (MPI_Comm_split)"; style=dashed; fontname="Helvetica"; fontsize=11;
    lead [label="LIDER\n1) clasifica su bloque de test con el\n   mejor K conocido (K inicial supuesto)\n2) cada 25 muestras: MPI_Iprobe\n3) tracker: cuando un K completa sus\n   folds, es el mejor hasta ahora?\n4) avisa de cada cambio de K"];
    work [label="procesos de evaluacion\n1) clasifican su bloque con el K vigente\n2) cada 25 muestras: MPI_Iprobe\n3) K nuevo -> DESCARTAN y REINICIAN\n4) K definitivo + bloque hecho -> fin"];
    help [shape=note, style="", label="MODO DINAMICO (paso 3):\nsi el bloque esta hecho y el ajuste\nno ha terminado, AYUDAN pidiendo\ntareas de ajuste al lider"];
    work -> help [style=dotted, arrowhead=none];
  }

  ta -> lead [label="MPI_Isend {tarea, accuracy}\n(TAG_RESULT, cada vez que acaba una tarea)", color="#1f77b4", fontcolor="#1f77b4"];
  lead -> ta [label="(dinamico) TAG_TASK {tarea | -1}\nrespuesta a TAG_REQ", style=dashed, color="#1f77b4", fontcolor="#1f77b4"];
  lead -> work [label="MPI_Isend {K, definitivo?}\n(TAG_BEST, cuando mejora el K;\nel ultimo con definitivo = 1)", color="#d62728", fontcolor="#d62728"];
  work -> lead [label="(dinamico) TAG_REQ y\nTAG_RESULT de la ayuda", style=dashed, color="#2ca02c", fontcolor="#2ca02c"];

  bar [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Ibarrier final (sin gastar CPU;\nel lider sigue atendiendo peticiones)"];
  red [shape=box, style="filled,rounded", fillcolor="#ffe9c7", label="MPI_Reduce(SUM) de aciertos\nen el comunicador del grupo\n+ MPI_Gather de metricas -> lider"];
  ta -> bar; lead -> bar; work -> bar;
  bar -> red;
  out [shape=oval, label="lider: Best k, Test accuracy,\nevolucion del mejor K, reinicios,\ntrabajo descartado, CSV"];
  red -> out;

  det [shape=note, style="", fontsize=9, label="K definitivo = mayor accuracy;\nen empate, el K menor\n(igual que el secuencial).\nResultado identico al secuencial."];
  det -> lead [style=dotted, arrowhead=none];
}
#@@END grafos/paso2_3_especulativo.dot
__PAYLOAD_NO_EJECUTAR__