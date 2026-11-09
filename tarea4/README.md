# Tarea 4 – kNN con MPI

Código de Carlos (repo `ccg155/Pr-ctica-3---knn2`, commit `b4024dd`), en `knn2/`:

| Fichero | Qué es |
|---|---|
| `knn_seq.cpp` | Secuencial parametrizado (referencia) |
| `knn_mpi.cpp` | Paso 1: paralelismo de datos |
| `knn_mpi_spec.cpp` | Pasos 2 y 3: funcional + especulativa (estática y dinámica) |
| `run_all.sh` | Compila, verifica, perfila, mide y dibuja. Lleva dentro los fuentes, `graficas.py` y los grafos `.dot` |

## Lanzarlo en varios nodos de la sala

En **todos** los equipos: `su usuariocp`. En el principal, **desde la carpeta personal de `usuariocp`**
(no desde `/home/alumno/...`: `distribute` crea la misma ruta en los demás nodos):

```sh
cd ~ && git clone https://github.com/gcg50-ua/IC_Prac3 && cd IC_Prac3/tarea4/knn2
mkdir -p build && cp -r <MNIST>/train <MNIST>/test build/     # o ./get_dataset.sh
nano hosts.txt                                                 # una IP 172.21.243.x por línea
ssh-keygen -t rsa                                              # Enter a todo
ssh-copy-id usuariocp@<IP>                                     # por cada secundario
HOSTFILE=hosts.txt SIZES="2000 4000 8000" REPS=3 bash run_all.sh
```

Con MPICH el script no usa `-mca plm...`. Si hay problemas de permisos: `chmod 777 -R <carpeta>`.
