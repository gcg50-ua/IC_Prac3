# Guía para medir las tareas 3 y 4 en la sala

Pensada para hacerlo **una sola persona**. Recoge lo que aprendimos en la tarea 2 y
los problemas del 9 de noviembre. Tiempo total: **unos 45 minutos** de ejecución.

El enunciado exige un **mínimo de 3 ordenadores**. Los del 9 de noviembre fueron:

| Equipo | IP | Papel |
|---|---|---|
| LS12I-20 | 172.21.243.170 | principal |
| LS12I-19 | 172.21.243.169 | secundario |
| LS12I-21 | 172.21.243.171 | secundario |

Si usas otros, cambia las IPs en `hosts_sala.txt` (una por línea, el principal incluido).

---

## 1. En cada secundario (1 minuto por equipo)

```sh
su usuariocp          # contraseña de usuariocp
hostname -I           # apunta la IP que empieza por 172.21.243.
```

Nada más. Deja la sesión abierta. **No uses la IP 172.17.0.1**: es la de Docker y es igual en todos.

## 2. En el principal

### 2.1. Usuario y carpeta (importante)

```sh
su usuariocp
cd ~
```

Trabaja **siempre en la carpeta personal de `usuariocp`** (`/home/usuariocp`), nunca en
`/home/alumno/...` ni en el pen. Los scripts crean **la misma ruta** en los otros nodos, y
`usuariocp` no puede escribir en `/home/alumno` (el 19 de octubre falló por esto).

### 2.2. Código y datos

```sh
git clone https://github.com/gcg50-ua/IC_Prac3     # o: cd ~/IC_Prac3 && git pull
cd ~/IC_Prac3
nano hosts_sala.txt                                 # comprueba las IPs
```

MNIST (solo para la tarea 4), desde el pen o desde el repo de Carlos:

```sh
mkdir -p tarea4/knn2/build
cp -r "<ruta>/knn2/build/train" "<ruta>/knn2/build/test" tarea4/knn2/build/
ls tarea4/knn2/build/train tarea4/knn2/build/test   # 2 archivos en cada una
```

Si `usuariocp` no puede leer el pen («Permiso denegado»): `exit`, copia como `alumno` a `/tmp`,
`su usuariocp` y copia de `/tmp` a `build/`. Si no hay MNIST, `run_all.sh` intenta
descargarlo (necesita red).

### 2.3. Comprobar el clúster (1 minuto)

```sh
bash comprobar_nodos.sh hosts_sala.txt
```

Tiene que decir `OK` para cada secundario y, al final, mostrar **los 3 nombres de equipo**.
Si es así, sigue. Si no, mira la sección 5.

> En la sala, `usuariocp` entra por SSH en los demás equipos **sin contraseña y sin claves**
> (la autenticación es «none»). **No hace falta `ssh-keygen` ni `ssh-copy-id`**: el 9 de
> noviembre `ssh-copy-id` se quedó colgado y no servía para nada.

## 3. Tarea 3: `defect_detection` (unos 15–20 min)

```sh
cd ~/IC_Prac3/tarea3
chmod +x *.sh
HOSTFILE=../hosts_sala.txt ./verificar_mpi.sh
```

Tiene que acabar con **«Todas las salidas coinciden con la versión secuencial»**. Si dice
`FALLO` o no compila, para y avisa (foto del error).

```sh
HOSTFILE=../hosts_sala.txt NPS="1 2 3 4 6 8 12" ./benchmark_mpi.sh
```

Deja un CSV en `tarea3/tiempos_mpi/`. Con prisa: añade `RONDAS=1` (unos 5 min).
Las gráficas (`python3 graficas_mpi.py tiempos_mpi/<csv>`) pueden fallar si no hay
matplotlib; da igual, se hacen luego con el CSV.

## 4. Tarea 4: kNN (unos 20–25 min con `REPS=1`)

**Cuando termine la tarea 3**, nunca a la vez: compartirían núcleos y los tiempos de las dos
saldrían mal.

```sh
cd ~/IC_Prac3/tarea4/knn2
HOSTFILE=../../hosts_sala.txt SIZES="2000 4000 8000" NPS="1 2 3 4 6 8 12 18 24" REPS=1 bash run_all.sh
```

- Pon `NPS` siempre: sin él, el script llega a 36 procesos y tarda mucho más.
- Con tiempo de sobra, `REPS=3` (alrededor de 1 hora).
- Al acabar, **abre `resultados/nodos.txt`**: tienen que salir los 3 equipos.
- `resultados/verificacion.txt` tiene que decir que las versiones MPI coinciden con la secuencial.

## 5. Problemas que ya nos han pasado

| Síntoma | Causa | Solución |
|---|---|---|
| `assert (!closed) failed` / `process manager error` | El ejecutable no está en la misma ruta en algún nodo | Trabajar en `~` como `usuariocp`; los scripts lo copian solos |
| `ssh` o `ssh-copy-id` se quedan colgados | No hacen falta en la sala | No usarlos; `comprobar_nodos.sh` tiene tiempo límite |
| `Ctrl+C` no responde | `ssh` colgado | `Ctrl+\` o cerrar la ventana de terminal (no se pierde nada) |
| `chmod: Operación no permitida`, `dubious ownership` | Archivos de `alumno` usados como `usuariocp` | Clonar el repo desde `~` de `usuariocp` |
| `Permission denied` al ejecutar `./script.sh` | Subido desde Windows sin permiso de ejecución | `chmod +x *.sh` o `bash script.sh` |
| Otros problemas de permisos | — | `chmod 777 -R <carpeta>` |
| `ls: no se puede acceder a 'build/train'` | Lanzado desde dentro de `build/` | `cd ~/IC_Prac3/tarea4/knn2` |
| `-mca plm_rsh_no_tree_spawn` del enunciado | Es de Open MPI; en la sala hay **MPICH** | No ponerlo (los scripts ya no lo usan) |
| `--map-by node`, `--oversubscribe` | También son de Open MPI | No hacen falta: MPICH reparte por turnos entre nodos |

Si `comprobar_nodos.sh` da `FALLO` en un nodo: comprueba la IP con `ping -c 3 <IP>` y, en
ese equipo, `systemctl status ssh` (tiene que poner `active (running)`). Si no se arregla,
usa otro ordenador.

## 6. Antes de irte: guardar los resultados

Los resultados están **solo en el principal** y se pierden al apagar. `usuariocp` no puede
escribir en el pen, así que se pasa por `/tmp`:

```sh
cp -r ~/IC_Prac3/tarea3/tiempos_mpi ~/IC_Prac3/tarea4/knn2/resultados /tmp/
exit                                                   # vuelves a ser alumno
cp -r /tmp/tiempos_mpi /tmp/resultados "/media/alumno/<PEN>/"
```

Lo que hay que traer:

- **Tarea 3:** la carpeta `tiempos_mpi/` (el CSV) y, si salieron, las gráficas.
- **Tarea 4:** la carpeta `resultados/` **entera** (resumen, CSV, gráficas, `nodos.txt`,
  `arquitectura.txt`, `verificacion.txt`, `gprof/`, `grafos/`, `logs/`).
- Anota qué 3 equipos usaste, si no son los de la tabla de arriba.
