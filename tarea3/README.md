# Tarea 3 – Paralelización con MPI de `defect_detection.cc`

| Fichero | Qué es |
|---|---|
| `defect_detection.cc` | Versión secuencial de la Práctica 2 (sin cambios) |
| `defect_detection_mpi.cc` | **Versión paralela MPI**, dos estrategias: `datos` y `tareas` |
| `verificar_mpi.sh` | Comprueba que la salida MPI es idéntica byte a byte a la secuencial |
| `benchmark_mpi.sh` | Mide secuencial + MPI (5 imágenes × estrategias × nº de procesos × rondas) → CSV |
| `graficas_mpi.py` | Genera las gráficas (speedup, eficiencia, tiempos, cómputo/comunicación) y `resumen.csv` |
| `benchmark.sh`, `LEEME.txt` | Benchmark de flags de la Práctica 2 (no cambian) |

## 1. Identificación (qué paralelizar)

Tiempos de la versión secuencial (`-O2`, imagen 2048×2048, N=10, acumulados en ms):

| Fase | ms | % |
|---|---:|---:|
| Lectura | 42 | 1,3 |
| **4 filtros** (blur 744, sharpen 734, sobel-X 659, sobel-Y 658) | **2795** | **85,1** |
| Combinaciones (suma 40 + pitágoras 117) | 157 | 4,8 |
| Marcado | 90 | 2,7 |
| Escritura | 197 | 6,0 |
| **Total** | **3284** | |

El perfil de gprof (tarea 1) apunta a lo mismo: casi todo el tiempo se va en
`convolve3x3` y en `Image::at` (se llama 9 veces por píxel y por filtro).
Las combinaciones y el marcado también son bucles por píxel, así que se
paralelizan junto a los filtros. Solo la lectura y la escritura quedan secuenciales.

## 2. Grafo de dependencias y paralelismo disponible

```
                       original
          ┌──────────┬────┴─────┬──────────┐
        blur      sharpen    sobel-X    sobel-Y      4 tareas independientes
          └────┬─────┘          └────┬─────┘
             suma                pitágoras           dependen de 2 filtros
               └─────────┬──────────┘
                      marcado                        depende de ambas ramas
```

- **Paralelismo funcional (de tareas):** los 4 filtros no dependen entre sí; las dos
  ramas (realzada / bordes) tampoco hasta el marcado.
- **Paralelismo de datos:** dentro de cada tarea cada píxel de salida se calcula
  de forma independiente (solo lee una ventana 3×3 de la entrada).

## 3. Estrategias implementadas

### `datos` (descomposición de dominio por bloques de filas) – la principal

```
P0 ─ Scatterv(bloques de filas) ─► P0 … Pp-1
     Sendrecv(fila halo) entre vecinos Pi ↔ Pi+1
     [cada Pi: 4 filtros + suma + pitágoras + marcado de SUS filas]
     Reduce(SUM nº píxeles sospechosos) ─► P0
     Gatherv(bordes), Gatherv(rgb) ─► P0          P0 escribe los ficheros
```

- Reparto equilibrado: `H/p` filas por proceso y las `H%p` sobrantes, una a cada uno de los primeros.
- El kernel 3×3 necesita la fila de arriba y la de abajo del bloque, así que
  cada proceso intercambia **una fila de halo** con cada vecino (`MPI_Sendrecv`, con
  `MPI_PROC_NULL` en los extremos). En el borde real de la imagen se replica la
  fila, igual que el `clamp` de la versión secuencial → resultado idéntico.
- Solo se recoge lo que P0 necesita escribir (bordes y rgb), no los intermedios.
- Comunicación por proceso y repetición: ~`W·H/p` bytes de ida, `2W` de halo,
  ~`4·W·H/p` de vuelta. El cómputo es ~`W·H/p`, así que la relación
  cómputo/comunicación no mejora con p: escala bien mientras la red no sea el cuello de botella.

### `tareas` (descomposición funcional siguiendo el grafo)

```
P0: Bcast ─► blur ─────────┐ suma ─────────────┐ marcado
P1: Bcast ─► sharpen ─Isend┘                   │
P2: Bcast ─► sobel-X ──────────┐ pitágoras ─Isend┘
P3: Bcast ─► sobel-Y ─Isend────┘
```

- Asignación con Q = min(p, 4): blur→0, sharpen→1%Q, sobel-X→2%Q, sobel-Y→3%Q;
  cada combinación va al proceso que ya tiene una de sus entradas, para mover menos datos.
- Todos los procesos con tarea necesitan la imagen entera (`MPI_Bcast` sobre un
  subcomunicador creado con `MPI_Comm_split`); los procesos con rango ≥ 4 no tienen trabajo.
- Envíos no bloqueantes (`MPI_Isend` + `MPI_Waitall`) para evitar interbloqueos.
- Límite: como mucho 4 procesos útiles, y el camino crítico (blur → suma → marcado) es secuencial.

### Funciones MPI utilizadas

| Función | Para qué |
|---|---|
| `MPI_Init`, `MPI_Comm_rank`, `MPI_Comm_size`, `MPI_Finalize` | Arranque/identificación/fin (modelo SPMD) |
| `MPI_Bcast` | Dimensiones de la imagen a todos; imagen completa en `tareas` |
| `MPI_Scatterv` | Reparto de bloques de filas de tamaño desigual (`datos`) |
| `MPI_Sendrecv` | Intercambio de halos sin interbloqueo (`datos`) |
| `MPI_Gatherv` | Recogida de los bloques de bordes y rgb en P0 (`datos`) |
| `MPI_Reduce` (SUM / MAX) | Total de píxeles sospechosos; tiempo del proceso más lento |
| `MPI_Isend` / `MPI_Recv` / `MPI_Waitall` | Aristas del grafo entre procesos (`tareas`) |
| `MPI_Comm_split`, `MPI_Comm_free` | Subcomunicador con los procesos que tienen tarea |
| `MPI_Barrier` | Todos empiezan a medir a la vez |
| `MPI_Wtime` | Medida de tiempos (cómputo vs. comunicación por proceso) |
| `MPI_Abort` | Si P0 no puede leer la imagen |

**Clasificación (tema 3):** modelo de paso de mensajes; estilo SPMD; estructura
maestro-esclavo (P0 reparte, recoge y hace E/S); en `datos` paralelismo de datos con
comunicación colectiva más punto a punto entre vecinos; en `tareas` paralelismo
funcional con comunicación punto a punto no bloqueante.

## 4. Cómo ejecutarlo

En local (cualquier Linux con Open MPI o MPICH):

```sh
./verificar_mpi.sh                         # debe decir que todas coinciden
mpic++ -O2 -std=c++17 -o defect_detection_mpi defect_detection_mpi.cc
mpirun -n 4 ./defect_detection_mpi entradas/engranaje2048x2048.pgm 60 datos
mpirun -n 4 ./defect_detection_mpi entradas/engranaje2048x2048.pgm 60 tareas
```

En la sala (equipo principal, como `usuariocp`, con el SSH ya preparado con
`tarea2/setup_mpi.sh` y `hosts.txt` actualizado):

```sh
cp -r tarea3 ~/tarea3 && cd ~/tarea3 && chmod +x *.sh
HOSTFILE=../tarea2/hosts.txt ./verificar_mpi.sh
HOSTFILE=../tarea2/hosts.txt NPS="1 2 3 4 6 8 12" ./benchmark_mpi.sh
python3 graficas_mpi.py tiempos_mpi/<csv generado>
```

- El script copia el ejecutable a la **misma ruta** en cada nodo. Las imágenes no hacen falta en los
  nodos secundarios: solo P0 lee y escribe.
- Si `mpirun` se queja de que no hay *slots* suficientes, poner `slots=<núcleos>` en cada línea
  del hostfile (p. ej. `172.21.243.178 slots=4`) o añadir `MPIRUN_EXTRA="--oversubscribe"`.
- `MPIRUN_EXTRA` pasa opciones extra a `mpirun` en los dos scripts. En la sala hay **MPICH**, que
  ya reparte los procesos por turnos entre los nodos del hostfile: no hace falta nada. Solo con
  Open MPI (que llena primero el primer equipo) conviene `MPIRUN_EXTRA="--map-by node"`.
- Para repartir los procesos entre máquinas en lugar de llenar una primero:
  `MPIRUN_EXTRA="--map-by node"`. Medir con ambos repartos sirve para la pregunta 3
  (comunicación dentro de un nodo vs. por la red).

## 5. Análisis teórico (para las preguntas de evaluación)

Con los tiempos de la sección 1 (N=10):

- **`datos` – Amdahl:** parte secuencial = lectura + escritura ≈ 239 ms de 3284 → f ≈ 7,3 %,
  así que S_max = 1/f ≈ 13,7. Con N=60 el cómputo pesa 6 veces más y f ≈ 1,3 % → S_max ≈ 77.
  En la práctica lo limita antes la comunicación: `Scatterv` y `Gatherv` pasan por P0, que envía y recibe
  ~5 bytes por píxel en cada repetición. Ese volumen no baja al subir p, mientras que el cómputo
  sí baja (∝ 1/p).
- **`tareas` – camino crítico:** max(blur + suma, sobel-X + pitágoras) + marcado ≈ max(784, 775) + 90
  ≈ 874 ms, más 239 de E/S → S_max ≈ 3284 / 1113 ≈ 2,95 con 4 procesos, sin contar la
  comunicación. Con más de 4 procesos no puede mejorar.
- **Speedup "por encima de lo esperado" con p=1:** la versión MPI reserva sus buffers una vez,
  mientras que la secuencial crea vectores nuevos en cada repetición (y usa `vector<bool>`).
  Por eso `graficas_mpi.py` dibuja también el **speedup relativo** T_MPI(1)/T_MPI(p), que mide solo el
  efecto del paralelismo (relacionado con la tarea 0: reservar el tamaño de antemano).
- La gráfica `4_computo_vs_comunicacion.png` responde la pregunta 3. La comunicación
  incluye el tiempo de espera (sincronización y desequilibrio de carga), porque se mide alrededor de
  cada llamada MPI bloqueante.
