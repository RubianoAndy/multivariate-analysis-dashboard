"""Genera el dataset simulado, la estadística descriptiva y las figuras en Python.

Rutas: el script se ubica en codes -> utils -> raíz del proyecto. Escribe el CSV
y los estadísticos en ``data/`` y las imágenes en
``public/assets/images/figures/python/``. El costo se expresa en miles de COP.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python"
GOOD_DIR = FIGURES_DIR / "good_design"
BAD_DIR = FIGURES_DIR / "bad_design"

for d in (DATA_DIR, PROCESSED_DIR, GOOD_DIR, BAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

rng = np.random.default_rng(42)

n = 120
sectors = rng.choice(["Residencial", "Comercial", "Industrial"], size=n, p=[0.5, 0.3, 0.2])
base = {"Residencial": 250, "Comercial": 900, "Industrial": 2500}
spread = {"Residencial": 60, "Comercial": 220, "Industrial": 600}
consumption = np.array([rng.normal(base[s], spread[s]) for s in sectors]).clip(50)
tariff = {"Residencial": 820, "Comercial": 710, "Industrial": 640}
cost = consumption * np.array([tariff[s] for s in sectors]) * rng.normal(1, 0.04, n) / 1000

df = pd.DataFrame({
    "cliente_id": [f"CL-{i:03d}" for i in range(1, n + 1)],
    "sector": sectors,
    "consumo_kwh": consumption.round(1),
    "costo_miles_cop": cost.round(1),
})
df.to_csv(DATA_DIR / "consumo_energia.csv", index=False)

stats = df.groupby("sector")["consumo_kwh"].agg(
    n="count", media="mean", mediana="median", desv_std="std", minimo="min", maximo="max"
).round(1)
stats.to_csv(PROCESSED_DIR / "stats_by_sector.csv")
print(stats)
corr = df["consumo_kwh"].corr(df["costo_miles_cop"])
print(f"Correlacion consumo-costo (Pearson): {corr:.3f}")
(PROCESSED_DIR / "corr.txt").write_text(f"{corr:.3f}")

"""GRÁFICAS BIEN DISEÑADAS (good_design).

Principios aplicados: título informativo, ejes con unidades, eje desde cero,
cuadrícula sutil, colores sobrios y etiquetas de datos.
"""

"""Histograma del consumo."""
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.hist(df["consumo_kwh"], bins=25, color="#2c7fb8", edgecolor="white")
ax.set_title("Distribución del consumo mensual de energía (n=120)")
ax.set_xlabel("Consumo (kWh/mes)")
ax.set_ylabel("Frecuencia (clientes)")
fig.tight_layout(); fig.savefig(GOOD_DIR / "hist_consumption.png"); plt.close(fig)

"""Barras: consumo promedio por sector.

El eje vertical arranca en cero y reserva un 15 % de holgura sobre la barra más
alta para que las etiquetas de datos no se solapen con el título.
"""
means = df.groupby("sector")["consumo_kwh"].mean().sort_values()
fig, ax = plt.subplots(figsize=(6.5, 3.4))
bars = ax.bar(means.index, means.values, color=["#a6bddb", "#74a9cf", "#2b8cbe"])
ax.bar_label(bars, fmt="%.0f", padding=3)
ax.set_ylim(0, means.max() * 1.15)
ax.set_title("Consumo promedio mensual por sector")
ax.set_xlabel("Sector"); ax.set_ylabel("Consumo promedio (kWh/mes)")
fig.tight_layout(); fig.savefig(GOOD_DIR / "bar_mean_consumption_by_sector.png"); plt.close(fig)

"""Dispersión + regresión: relación entre consumo y costo, con color por sector."""
fig, ax = plt.subplots(figsize=(6.5, 3.8))
colors = {"Residencial": "#1b9e77", "Comercial": "#d95f02", "Industrial": "#7570b3"}
for s, g in df.groupby("sector"):
    ax.scatter(g["consumo_kwh"], g["costo_miles_cop"], s=18, alpha=0.7, label=s, color=colors[s])
m, b = np.polyfit(df["consumo_kwh"], df["costo_miles_cop"], 1)
xs = np.linspace(df["consumo_kwh"].min(), df["consumo_kwh"].max(), 100)
ax.plot(xs, m * xs + b, "k--", lw=1, label=f"Tendencia (r = {corr:.2f})")
ax.set_title("Relación entre consumo y costo mensual")
ax.set_xlabel("Consumo (kWh/mes)"); ax.set_ylabel("Costo (miles de COP)")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(GOOD_DIR / "scatter_consumption_vs_cost.png"); plt.close(fig)

"""Boxplot por sector: dispersión del consumo dentro de cada grupo."""
fig, ax = plt.subplots(figsize=(6.5, 3.6))
order = ["Residencial", "Comercial", "Industrial"]
data = [df.loc[df["sector"] == s, "consumo_kwh"] for s in order]
bp = ax.boxplot(data, tick_labels=order, patch_artist=True, medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], ["#a6bddb", "#74a9cf", "#2b8cbe"]):
    patch.set_facecolor(c)
ax.set_title("Dispersión del consumo por sector (diagrama de caja)")
ax.set_xlabel("Sector"); ax.set_ylabel("Consumo (kWh/mes)")
fig.tight_layout(); fig.savefig(GOOD_DIR / "boxplot_consumption_by_sector.png"); plt.close(fig)

"""Barras horizontales: versión CORRECTA de la torta.

Compara longitudes sobre un eje común de 0 a 100 % en lugar de ángulos.
"""
share = df.groupby("sector")["consumo_kwh"].sum()
share_pct = (share / share.sum() * 100).sort_values()
fig, ax = plt.subplots(figsize=(6.0, 2.9))
bars = ax.barh(share_pct.index, share_pct.values, color="#2b8cbe")
ax.bar_label(bars, fmt="%.1f%%", padding=3)
ax.set_title("Participación por sector en el consumo total")
ax.set_xlabel("Participación (%)"); ax.set_xlim(0, 100)
fig.tight_layout(); fig.savefig(GOOD_DIR / "barh_sector_share.png"); plt.close(fig)

"""GRÁFICA MAL DISEÑADA (bad_design), para análisis crítico.

Errores intencionales: título vago sin unidades ni etiquetas de datos, colores
saturados sin función, leyenda separada, sombras y explode que distorsionan las
áreas, y ángulo de inicio arbitrario.
"""

"""Torta mal diseñada: versión INCORRECTA."""
fig, ax = plt.subplots(figsize=(5.2, 3.6))
explode = (0.08, 0.08, 0.08)
wild = ["#ff00ff", "#00ffff", "#ffff00"]
ax.pie(share.values, labels=None, colors=wild, explode=explode, shadow=True, startangle=17)
ax.legend(share.index, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
ax.set_title("Consumo", fontsize=10)
fig.tight_layout(); fig.savefig(BAD_DIR / "pie_sector_share_bad.png"); plt.close(fig)

print("OK - dataset, estadísticos y figuras de Python generados")
