# Tarea 1 – gprof

Se perfilan **dos** códigos: el nuestro de la práctica anterior (`../tarea3/defect_detection.cc`)
y el proporcionado (`../tarea4/knn2/`). Hay que hacerlo en Linux (gprof no funciona en macOS).

```sh
g++ -O0 -g -pg programa.cc -o programa   # repetir también con -O2
./programa
gprof -b -p -q ./programa gmon.out > perfil.txt
```

Guardar aquí los perfiles (`-O0` y `-O2`) de cada código.
