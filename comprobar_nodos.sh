#!/usr/bin/env bash
# Comprueba que el clúster está listo ANTES de lanzar las medidas.
# Uso (en el equipo principal, como usuariocp):
#   bash comprobar_nodos.sh hosts.txt
# Para cada IP: responde por SSH sin contraseña (con tiempo límite, para que
# no se quede colgado). Después lanza "hostname" con mpirun en todos los nodos.
set -uo pipefail

HOSTS="${1:-hosts.txt}"
[ -f "$HOSTS" ] || { echo "No existe $HOSTS"; exit 1; }

echo "== Usuario: $(whoami)   Equipo: $(hostname)   Carpeta: $PWD"
[ "$(whoami)" = "usuariocp" ] || echo "   AVISO: no eres usuariocp (haz: su usuariocp)"
case "$PWD" in /home/usuariocp*) ;; *) echo "   AVISO: no estás en la carpeta personal de usuariocp (haz: cd ~)" ;; esac

MIS_IPS=" $(hostname -I) "
fallos=0
n=0
echo "== SSH a cada nodo de $HOSTS"
while read -r ip _; do
    ip="${ip%%$'\r'}"
    [ -z "$ip" ] && continue
    [[ "$ip" == \#* ]] && continue
    n=$((n + 1))
    if [[ "$MIS_IPS" == *" $ip "* ]]; then
        echo "   $ip  este equipo ($(hostname))"
        continue
    fi
    nombre=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$ip" hostname 2>&1)
    if [ $? -eq 0 ]; then
        echo "   $ip  OK ($nombre)"
    else
        echo "   $ip  FALLO: $nombre"
        fallos=$((fallos + 1))
    fi
done < "$HOSTS"

[ "$fallos" -eq 0 ] || { echo "== Hay $fallos nodo(s) sin SSH: arréglalo antes de seguir (ver GUIA_SALA.md)"; exit 1; }

echo "== mpirun en los $n nodos (cada línea es un proceso)"
mpirun -hostfile "$HOSTS" -n "$((n * 2))" hostname | sort | uniq -c
echo "== Si arriba salen los $n equipos, el clúster está listo."
