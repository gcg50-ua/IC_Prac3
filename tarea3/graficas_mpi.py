#!/usr/bin/env python3
"""Graficas de la tarea 3 a partir del CSV de benchmark_mpi.sh.

Uso:  python3 graficas_mpi.py tiempos_mpi/tiempos_mpiLinuxAAAAMMDD-HHMM.csv
Necesita matplotlib (pip install matplotlib). Deja los PNG en graficas_mpi/ y
un resumen (mediana de las rondas, speedup y eficiencia) en
graficas_mpi/resumen.csv.

Definiciones usadas:
  T1       = tiempo total de la version SECUENCIAL original (-O2)
  Tp       = tiempo total de la version MPI con p procesos
  speedup  S(p) = T1 / Tp
  eficiencia E(p) = S(p) / p
  speedup relativo Srel(p) = T_MPI(1) / T_MPI(p)  (misma estrategia; aisla el
  efecto del paralelismo de las mejoras de la version MPI con 1 proceso, que
  reserva la memoria una sola vez en lugar de en cada repeticion)
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paleta categorica validada (5 tamanos de imagen) + marcadores distintos
# para que la identidad no dependa solo del color.
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARCAS = ["o", "s", "^", "D", "v"]
TINTA = "#2b2b29"
TINTA_SUAVE = "#6b6a63"
REJILLA = "#e4e3dd"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": TINTA_SUAVE,
    "axes.labelcolor": TINTA,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "axes.grid": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 6,
})


def leer(ruta):
    """Devuelve {(estrategia, ancho, procesos): {columna: mediana}}."""
    grupos = defaultdict(list)
    with open(ruta, newline="") as f:
        for fila in csv.DictReader(f):
            if not fila.get("total_ms"):
                continue
            clave = (fila["estrategia"], int(fila["ancho_px"]), int(fila["procesos"]))
            grupos[clave].append(fila)
    datos = {}
    for clave, filas in grupos.items():
        datos[clave] = {
            col: statistics.median(float(r[col]) for r in filas)
            for col in filas[0]
            if col.endswith("_ms")
        }
    return datos


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    datos = leer(sys.argv[1])
    salida = "graficas_mpi"
    os.makedirs(salida, exist_ok=True)

    anchos = sorted({a for (_, a, _) in datos})
    estrategias = [e for e in ("datos", "tareas") if any(k[0] == e for k in datos)]
    T1 = {a: datos[("secuencial", a, 1)]["total_ms"] for a in anchos if ("secuencial", a, 1) in datos}

    # ---- resumen.csv ----
    with open(os.path.join(salida, "resumen.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estrategia", "ancho_px", "procesos", "total_ms", "speedup", "speedup_relativo", "eficiencia",
                    "computo_medio_ms", "comunicacion_media_ms", "pct_comunicacion"])
        for (e, a, p), v in sorted(datos.items()):
            s = T1[a] / v["total_ms"] if a in T1 else float("nan")
            cm, co = v["computo_medio_ms"], v["comunicacion_media_ms"]
            pct = 100 * co / (cm + co) if cm + co > 0 else 0
            base = datos.get((e, a, 1), v)["total_ms"]
            w.writerow([e, a, p, f"{v['total_ms']:.1f}", f"{s:.3f}",
                        f"{base / v['total_ms']:.3f}", f"{s / p:.3f}",
                        f"{cm:.1f}", f"{co:.1f}", f"{pct:.1f}"])

    def procesos(e, a):
        return sorted(p for (ee, aa, p) in datos if ee == e and aa == a)

    def figura_por_estrategia(nombre, titulo, etiqueta_y, valor, ideal=None, logy=False):
        fig, ejes = plt.subplots(1, len(estrategias), figsize=(5.5 * len(estrategias), 4.2),
                                 sharey=True, squeeze=False)
        for ax, e in zip(ejes[0], estrategias):
            pmax = 1
            for i, a in enumerate(anchos):
                ps = procesos(e, a)
                if not ps or a not in T1:
                    continue
                ys = [valor(e, a, p) for p in ps]
                ax.plot(ps, ys, color=COLORES[i % 5], marker=MARCAS[i % 5],
                        markeredgecolor="white", markeredgewidth=1.2, label=f"{a}x{a}")
                pmax = max(pmax, ps[-1])
            if ideal:
                xs = list(range(1, pmax + 1))
                ax.plot(xs, [ideal(x) for x in xs], color=TINTA_SUAVE, linestyle="--",
                        linewidth=1.2, label="ideal")
            ax.set_title(f"Estrategia: {e}", color=TINTA, loc="left")
            ax.set_xlabel("Numero de procesos MPI (p)")
            if logy:
                ax.set_yscale("log")
        ejes[0][0].set_ylabel(etiqueta_y)
        ejes[0][-1].legend(title="Imagen", loc="best")
        fig.suptitle(titulo, color=TINTA, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(os.path.join(salida, nombre), dpi=150)
        plt.close(fig)

    sp = lambda e, a, p: T1[a] / datos[(e, a, p)]["total_ms"]  # noqa: E731
    figura_por_estrategia("1_speedup.png", "Speedup S(p) = T_secuencial / T_p", "Speedup",
                          sp, ideal=lambda x: x)
    figura_por_estrategia("1b_speedup_relativo.png",
                          "Speedup relativo Srel(p) = T_MPI(1) / T_MPI(p)", "Speedup relativo",
                          lambda e, a, p: datos[(e, a, 1)]["total_ms"] / datos[(e, a, p)]["total_ms"],
                          ideal=lambda x: x)
    figura_por_estrategia("2_eficiencia.png", "Eficiencia E(p) = S(p) / p", "Eficiencia",
                          lambda e, a, p: sp(e, a, p) / p, ideal=lambda x: 1.0)
    figura_por_estrategia("3_tiempo_total.png", "Tiempo total de ejecucion",
                          "Tiempo (ms, escala log)",
                          lambda e, a, p: datos[(e, a, p)]["total_ms"], logy=True)

    # ---- 4: computo vs comunicacion (media por proceso), imagen mas grande ----
    a = anchos[-1]
    fig, ejes = plt.subplots(1, len(estrategias), figsize=(5.5 * len(estrategias), 4.2),
                             sharey=True, squeeze=False)
    for ax, e in zip(ejes[0], estrategias):
        ps = procesos(e, a)
        x = list(range(len(ps)))
        comp = [datos[(e, a, p)]["computo_medio_ms"] for p in ps]
        comm = [datos[(e, a, p)]["comunicacion_media_ms"] for p in ps]
        ax.bar(x, comp, width=0.6, color=COLORES[0], label="Computo", edgecolor="white",
               linewidth=2)
        ax.bar(x, comm, width=0.6, bottom=comp, color=COLORES[1],
               label="Comunicacion + espera", edgecolor="white", linewidth=2)
        for xi, c, m in zip(x, comp, comm):
            if c + m > 0:
                ax.text(xi, c + m, f"{100 * m / (c + m):.0f}%", ha="center", va="bottom",
                        fontsize=8, color=TINTA_SUAVE)
        ax.set_xticks(x, [str(p) for p in ps])
        ax.set_title(f"Estrategia: {e}", color=TINTA, loc="left")
        ax.set_xlabel("Numero de procesos MPI (p)")
        ax.grid(axis="x", visible=False)
    ejes[0][0].set_ylabel("Tiempo medio por proceso (ms)")
    ejes[0][-1].legend(loc="upper right")
    fig.suptitle(f"Reparto computo / comunicacion - imagen {a}x{a} "
                 "(% = fraccion de comunicacion)", color=TINTA, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "4_computo_vs_comunicacion.png"), dpi=150)
    plt.close(fig)

    # ---- 5: speedup frente al tamano del problema (p maximo de cada estrategia) ----
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for i, e in enumerate(estrategias):
        pmax = max(p for (ee, _, p) in datos if ee == e)
        xs = [aa for aa in anchos if (e, aa, pmax) in datos and aa in T1]
        ax.plot(xs, [sp(e, aa, pmax) for aa in xs], color=COLORES[i], marker=MARCAS[i],
                markeredgecolor="white", markeredgewidth=1.2, label=f"{e} (p={pmax})")
    ax.set_xlabel("Lado de la imagen (px)")
    ax.set_ylabel("Speedup")
    ax.set_xticks(anchos)
    ax.legend()
    ax.set_title("Speedup frente al tamano del problema", color=TINTA, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(salida, "5_speedup_vs_tamano.png"), dpi=150)
    plt.close(fig)

    print(f"Graficas y resumen.csv en {salida}/")


if __name__ == "__main__":
    main()
