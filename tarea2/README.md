# Tarea 2 – Ejecución distribuida MPI (en la sala de prácticas)

```sh
mpicc hola_mpi.c -o hola_mpi
mpirun -hostfile <archivo_hosts> -n <num_procesos> ./hola_mpi
```

Notas de clase:
- Usuario en todas las máquinas: `usuariocp`.
- **No** poner `-mca plm_rsh_no_tree_spawn 1` en `mpirun`.
- Si hay problemas de permisos con archivos: `chmod 777 -R <carpeta>`.
