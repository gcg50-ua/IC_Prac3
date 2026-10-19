#!/usr/bin/env bash
# Prepara y lanza el "hola mundo" MPI en varios equipos de la sala.
# Ejecutar en el equipo PRINCIPAL, como usuariocp:
#   su usuariocp
#   bash setup_mpi.sh [num_procesos]
# Lee las IPs de hosts.txt (junto a este script), una por linea.
set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$HOME/tarea2"          # misma ruta en todos los equipos
NP="${1:-6}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

[ "$(whoami)" = "usuariocp" ] || echo "AVISO: no eres usuariocp (eres $(whoami))."
[ -f "$SRC_DIR/hosts.txt" ] || { echo "Falta $SRC_DIR/hosts.txt"; exit 1; }

echo "== 1. Copiando archivos a $WORKDIR y compilando"
mkdir -p "$WORKDIR"
cp "$SRC_DIR/hola_mpi.c" "$SRC_DIR/hosts.txt" "$WORKDIR/"
cd "$WORKDIR"
sed -i 's/\r$//; /^[[:space:]]*$/d' hosts.txt   # quita CRLF y lineas vacias
chmod 600 hosts.txt
mpicc hola_mpi.c -o hola_mpi
chmod o+x hola_mpi

echo "== 2. Clave SSH"
[ -f "$HOME/.ssh/id_rsa" ] || ssh-keygen -t rsa -N "" -f "$HOME/.ssh/id_rsa"

MIS_IPS=" $(hostname -I) "
for ip in $(cat hosts.txt); do
  if [[ "$MIS_IPS" == *" $ip "* ]]; then
    echo "-- $ip es este equipo, se salta"
    continue
  fi
  echo "== 3. Secundario $ip"
  if ! ssh "${SSH_OPTS[@]}" -o BatchMode=yes "$ip" true 2>/dev/null; then
    echo "   Copiando clave (pedira la contrasena de usuariocp una vez)"
    ssh-copy-id "${SSH_OPTS[@]}" "usuariocp@$ip"
  fi
  ssh "${SSH_OPTS[@]}" "$ip" "mkdir -p '$WORKDIR'"
  scp "${SSH_OPTS[@]}" hola_mpi hosts.txt "$ip:$WORKDIR/"
  echo "   OK: $(ssh "$ip" "hostname; ls '$WORKDIR'" | tr '\n' ' ')"
done

echo "== 4. Ejecutando: mpirun -hostfile hosts.txt -n $NP ./hola_mpi"
mpirun -hostfile hosts.txt -n "$NP" ./hola_mpi
