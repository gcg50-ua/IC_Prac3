# Tarea 2 – Ejecución distribuida MPI (en la sala de prácticas)

```sh
mpicc hola_mpi.c -o hola_mpi
mpirun -hostfile <archivo_hosts> -n <num_procesos> ./hola_mpi
```

Notas de clase:
- Usuario en todas las máquinas: `usuariocp`.
- **No** poner `-mca plm_rsh_no_tree_spawn 1` en `mpirun`.
- Si hay problemas de permisos con archivos: `chmod 777 -R <carpeta>`.

## Script automático

En el equipo principal, como `usuariocp` (editar antes `hosts.txt` con las IPs del día):

```sh
bash setup_mpi.sh        # o: bash setup_mpi.sh 9  (número de procesos)
```

Copia todo a `~/tarea2`, compila, configura SSH sin contraseña con los secundarios,
copia el ejecutable a la misma ruta en cada uno y lanza `mpirun`.
