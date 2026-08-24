"""Actividad 6 - Fase 3: visualizacion avanzada con Seaborn.

Las figuras de la Fase 2 responden preguntas del modelo (cuantas componentes,
cuantos grupos). Esta fase responde preguntas del analista: como se relacionan
las variables entre si, en que se diferencian los grupos descubiertos y donde
falla la particion.

Seaborn se usa aqui -y no Matplotlib directamente- por tres capacidades que en
Matplotlib exigirian decenas de lineas: el mapa de calor anotado con paleta
divergente centrada, la matriz de dispersion por grupos (``PairGrid``) y el
mapa de calor con dendrogramas en los margenes (``clustermap``), que reordena
filas y columnas segun su similitud.

Requiere haber ejecutado antes ``pca_clustering.py``, porque lee las etiquetas
de cluster desde ``data/processed/clientes_con_cluster.csv``.

Ejecucion (desde la raiz del proyecto):
    python utils/codes/python/advanced_viz.py
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import VARIABLES_NUMERICAS
from pca_clustering import VARIABLES_LOG, preparar_matriz
from estilo import (
    aplicar_estilo_matplotlib,
    AZUL_UNISALLE, TEXTO, TEXTO_SUAVE, BORDE,
    COLOR_SECTOR, ORDEN_SECTOR, COLOR_CLUSTER,
    ESCALA_DIVERGENTE, ETIQUETAS_CORTAS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLUSTERS_PATH = PROCESSED_DIR / "clientes_con_cluster.csv"
FIGURAS_DIR = PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python" / "advanced"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

aplicar_estilo_matplotlib()
sns.set_context("notebook", font_scale=0.95)

# Posicion (izquierda, abajo, ancho, alto) de la barra de color del clustermap,
# en coordenadas de figura. A media altura y pegada al margen: arriba chocaria
# con el titulo y con el dendrograma de columnas.
POSICION_BARRA_COLOR = (0.015, 0.40, 0.022, 0.15)


def titulo(fig, texto, subtitulo=None):
    """Coloca titulo y subtitulo alineados a la izquierda y reserva su espacio.

    Las coordenadas de figura son fracciones de la altura total, asi que una
    separacion fija (0.04, por ejemplo) equivale a 0.34 pulgadas en una figura
    de 8.5 y a 0.14 en una de 3.6: en las bajas el subtitulo se monta sobre el
    titulo. Aqui las distancias se fijan en pulgadas y se convierten despues.

    Retorna
    -------
    float
        Fraccion de altura por debajo de la cual queda libre el lienzo; sirve
        como ``top`` del ``rect`` de ``tight_layout``.
    """
    ancho, alto = fig.get_size_inches()
    y_titulo = 1 - 0.24 / alto
    fig.suptitle(texto, fontsize=13, fontweight="bold", color=AZUL_UNISALLE,
                 x=0.012, ha="left", y=y_titulo)

    if not subtitulo:
        return y_titulo - 0.12 / alto

    # El subtitulo se parte en lineas segun el ancho real disponible.
    lineas = textwrap.wrap(subtitulo, width=int(ancho * 13.5))
    fig.text(
        0.012, y_titulo - 0.26 / alto, "\n".join(lineas),
        fontsize=9.5, color=TEXTO_SUAVE, ha="left", va="top", linespacing=1.35,
    )
    alto_subtitulo = 0.26 + 0.19 * len(lineas) + 0.10
    return y_titulo - alto_subtitulo / alto


# -----------------------------------------------------------------------------
# 1. MAPA DE CALOR DE CORRELACIONES
# -----------------------------------------------------------------------------
def heatmap_correlacion(df, ruta):
    """Matriz de correlacion anotada, con mascara triangular.

    Se calcula sobre las mismas variables transformadas que alimentan el PCA:
    una correlacion de Pearson sobre variables lognormales mide la relacion
    equivocada, porque Pearson supone linealidad y aqui la relacion es lineal
    solo en logaritmos.
    """
    datos = df[VARIABLES_NUMERICAS].copy()
    for col in VARIABLES_LOG:
        datos[col] = np.log(datos[col])
    datos.columns = [
        f"log {ETIQUETAS_CORTAS[c]}" if c in VARIABLES_LOG else ETIQUETAS_CORTAS[c]
        for c in datos.columns
    ]

    R = datos.corr()
    mascara = np.triu(np.ones_like(R, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    sns.heatmap(
        R, mask=mascara, annot=True, fmt=".2f", annot_kws={"size": 8},
        cmap=ESCALA_DIVERGENTE, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.6, linecolor="white",
        cbar_kws={"shrink": 0.72, "label": "Correlacion de Pearson (r)"}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=42, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)
    ax.grid(False)

    top = titulo(
        fig,
        "Matriz de correlacion entre las variables del cliente",
        "Dos bloques compactos: las seis variables de tamano y las tres de estado de la red. "
        "La temperatura no se asocia con ninguna.",
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. MATRIZ DE DISPERSION POR CLUSTER
# -----------------------------------------------------------------------------
def matriz_dispersion(df, ruta):
    """PairGrid de las variables representativas de cada bloque, por cluster.

    Se eligen cinco variables -dos de tamano, tres de estado de red- en vez de
    las nueve: una matriz 9x9 son 81 paneles ilegibles, y las variables del
    mismo bloque son casi redundantes entre si.
    """
    seleccion = [
        "consumo_kwh",
        "potencia_instalada_kw",
        "factor_potencia",
        "antiguedad_anios",
        "interrupciones_mes",
    ]
    datos = df[seleccion + ["cluster"]].copy()
    datos["consumo_kwh"] = np.log(datos["consumo_kwh"])
    datos["potencia_instalada_kw"] = np.log(datos["potencia_instalada_kw"])
    datos = datos.rename(
        columns={
            "consumo_kwh": "log Consumo",
            "potencia_instalada_kw": "log Potencia",
            "factor_potencia": "F. potencia",
            "antiguedad_anios": "Antiguedad",
            "interrupciones_mes": "Interrupciones",
        }
    )
    datos["Cluster"] = datos["cluster"].map(lambda c: f"C{c}")
    datos = datos.drop(columns="cluster")

    paleta = {f"C{c}": COLOR_CLUSTER[c % len(COLOR_CLUSTER)]
              for c in sorted(df["cluster"].unique())}

    g = sns.PairGrid(datos, hue="Cluster", hue_order=sorted(paleta), palette=paleta,
                     diag_sharey=False, height=1.75, corner=True)
    g.map_lower(sns.scatterplot, s=16, alpha=0.7, edgecolor="none")
    g.map_diag(sns.kdeplot, fill=True, alpha=0.45, linewidth=1.2)
    g.add_legend(title="Cluster", bbox_to_anchor=(0.97, 0.72), loc="upper right")

    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(labelsize=7.5)
            ax.xaxis.label.set_size(8.5)
            ax.yaxis.label.set_size(8.5)

    top = titulo(
        g.figure,
        "Matriz de dispersion de las variables representativas, por cluster",
        "Los cuatro grupos se separan limpiamente en los pares que cruzan tamano "
        "con estado de la red, no dentro de un mismo bloque.",
    )
    g.figure.subplots_adjust(top=top)
    g.figure.savefig(ruta)
    plt.close(g.figure)


# -----------------------------------------------------------------------------
# 3. CLUSTERMAP
# -----------------------------------------------------------------------------
def clustermap(df, ruta):
    """Mapa de calor de los datos estandarizados con dendrogramas en los margenes.

    Reordena clientes y variables por similitud, de modo que la estructura de
    bloques aparece sin imponerla: las bandas horizontales son los grupos y las
    columnas se agrupan solas en tamano frente a estado de la red.
    """
    X, nombres, _ = preparar_matriz(df)
    matriz = pd.DataFrame(
        X,
        columns=[
            f"log {ETIQUETAS_CORTAS[n.replace('log_', '')]}" if n.startswith("log_")
            else ETIQUETAS_CORTAS[n]
            for n in nombres
        ],
    )

    colores_fila = df["cluster"].map(
        lambda c: COLOR_CLUSTER[c % len(COLOR_CLUSTER)]
    ).rename("Cluster")

    g = sns.clustermap(
        matriz,
        method="ward",
        cmap=ESCALA_DIVERGENTE,
        center=0,
        vmin=-2.5,
        vmax=2.5,
        row_colors=colores_fila.values,
        yticklabels=False,
        figsize=(9.5, 8.5),
        dendrogram_ratio=(0.14, 0.10),
        cbar_pos=POSICION_BARRA_COLOR,
        cbar_kws={"label": "Valor estandarizado (z)"},
        linewidths=0,
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=40, ha="right", fontsize=8.5
    )
    g.ax_heatmap.set_ylabel("300 clientes reordenados por similitud", fontsize=9)
    g.ax_heatmap.grid(False)

    top = titulo(
        g.figure,
        "Clustermap: clientes y variables reordenados por similitud",
        "La franja de color a la izquierda es el cluster asignado por K-Means; coincide "
        "con los bloques que el dendrograma forma por su cuenta.",
    )
    # clustermap monta sus ejes sobre un gridspec propio, que ignora tight_layout;
    # hay que reducirle el techo a mano para dejar sitio al titulo. La barra de
    # color tambien pertenece a ese gridspec (celda [0, 0]) y seaborn la coloca
    # despues con set_position, asi que update() la devuelve a la esquina: se
    # reposiciona de nuevo tras mover la rejilla.
    g.gs.update(top=top)
    g.ax_cbar.set_position(POSICION_BARRA_COLOR)
    g.figure.savefig(ruta, dpi=150)
    plt.close(g.figure)


# -----------------------------------------------------------------------------
# 4. PERFIL DE LOS CLUSTERES
# -----------------------------------------------------------------------------
def perfil_clusters(df, ruta):
    """Mapa de calor de las medias por cluster en puntuaciones z.

    Es la tabla de interpretacion convertida en figura: cada celda dice cuantas
    desviaciones tipicas se aparta el cluster de la media global en esa
    variable. Sirve para nombrar los grupos de un vistazo.
    """
    media = df[VARIABLES_NUMERICAS].mean()
    desv = df[VARIABLES_NUMERICAS].std()
    perfil = (df.groupby("cluster")[VARIABLES_NUMERICAS].mean() - media) / desv
    perfil.columns = [ETIQUETAS_CORTAS[c] for c in perfil.columns]
    tamanos = df["cluster"].value_counts().sort_index()
    perfil.index = [f"C{c}\n(n={tamanos[c]})" for c in perfil.index]

    fig, ax = plt.subplots(figsize=(11, 3.6))
    sns.heatmap(
        perfil, annot=True, fmt="+.2f", annot_kws={"size": 8.5},
        cmap=ESCALA_DIVERGENTE, center=0, vmin=-1.5, vmax=1.5,
        linewidths=0.8, linecolor="white",
        cbar_kws={"label": "Desviaciones tipicas respecto a la media global",
                  "shrink": 0.85}, ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=28, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_ylabel("")
    ax.grid(False)

    top = titulo(
        fig,
        "Perfil de cada cluster en puntuaciones z",
        "La temperatura permanece plana en los cuatro grupos: la particion no reproduce "
        "la geografia, sino el tamano y el estado de la red.",
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 5. DISTRIBUCIONES POR CLUSTER
# -----------------------------------------------------------------------------
def distribuciones_cluster(df, ruta):
    """Violines de las dos dimensiones del modelo, con la caja y los datos dentro.

    El violin muestra la forma de la distribucion -no solo los cuartiles- y deja
    ver si un grupo es realmente compacto o si el promedio esconde dos modas.
    """
    variables = [
        ("consumo_kwh", "Consumo mensual (kWh, escala log)", True),
        ("factor_potencia", "Factor de potencia", False),
        ("antiguedad_anios", "Antiguedad de la instalacion (anios)", False),
        ("interrupciones_mes", "Interrupciones al mes", False),
    ]
    orden = sorted(df["cluster"].unique())
    paleta = [COLOR_CLUSTER[c % len(COLOR_CLUSTER)] for c in orden]
    etiquetas_x = [f"C{c}" for c in orden]

    fig, ejes = plt.subplots(1, 4, figsize=(14, 4.3))
    for ax, (col, titulo_eje, log) in zip(ejes, variables):
        sns.violinplot(
            data=df, x="cluster", y=col, hue="cluster", order=orden,
            palette=paleta, legend=False, inner="box", cut=0, linewidth=1, ax=ax,
        )
        sns.stripplot(
            data=df, x="cluster", y=col, order=orden, size=2.2,
            color=TEXTO, alpha=0.35, jitter=0.22, ax=ax,
        )
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel(titulo_eje, fontsize=9)
        ax.set_xticks(range(len(orden)))
        ax.set_xticklabels(etiquetas_x)
        ax.tick_params(labelsize=8.5)

    top = titulo(
        fig,
        "Distribucion de las variables clave dentro de cada cluster",
        "C0 y C3 (red heredada) concentran las interrupciones y la antiguedad; "
        "C1 y C3 concentran el consumo.",
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. REGRESION: EL EFECTO CLIMATICO QUE EL CLUSTERING NO CAPTURA
# -----------------------------------------------------------------------------
def regresion_temperatura(df, ruta):
    """Ajustes de regresion de consumo sobre temperatura, por sector.

    Version grafica de la ANCOVA de la Fase 1. Las tres rectas suben, pero lo
    hacen poco -unas 0.2 unidades de log a lo largo de veinte grados- frente a
    una dispersion interna que ronda media unidad y a una separacion entre
    sectores de cuatro unidades de logaritmo, unas cincuenta veces en kWh. Ese
    contraste de magnitudes es exactamente lo
    que impide a la t de Welch declarar significativa la diferencia entre
    regiones: el efecto existe, pero es pequeno comparado con el ruido que hay
    que atravesar para verlo sin controlar la escala del cliente.
    """
    datos = df.assign(log_consumo=np.log(df["consumo_kwh"]))

    g = sns.lmplot(
        data=datos, x="temperatura_c", y="log_consumo", hue="sector",
        hue_order=ORDEN_SECTOR, palette=COLOR_SECTOR, height=5.2, aspect=1.55,
        scatter_kws={"s": 26, "alpha": 0.65, "edgecolor": "none"},
        line_kws={"linewidth": 2},
        ci=95, legend=False,
    )
    ax = g.ax
    # Ajuste global sin distinguir sector: la recta plana del conjunto.
    sns.regplot(
        data=datos, x="temperatura_c", y="log_consumo", scatter=False,
        color=TEXTO_SUAVE, line_kws={"linewidth": 2.2, "linestyle": "--"},
        ci=None, ax=ax,
    )
    ax.set_xlabel("Temperatura media del municipio (C)")
    ax.set_ylabel("Consumo mensual, ln(kWh)")
    # Con fondo: la leyenda cae sobre la nube de puntos y sin recuadro no se lee.
    ax.legend(
        handles=[
            plt.Line2D([], [], color=COLOR_SECTOR[s], linewidth=2.4, label=s)
            for s in ORDEN_SECTOR
        ]
        + [plt.Line2D([], [], color=TEXTO_SUAVE, linewidth=2.2, linestyle="--",
                      label="Ajuste global (sin distinguir sector)")],
        fontsize=9, loc="lower right", frameon=True, framealpha=0.94,
        edgecolor=BORDE, facecolor="white",
    )

    top = titulo(
        g.figure,
        "Consumo frente a temperatura, dentro de cada sector",
        "Las cuatro pendientes son positivas, pero suaves: el efecto que la ANCOVA "
        "cuantifica en +1.06 % por grado queda pequeno frente a la dispersion interna "
        "de cada sector y minusculo frente a las cuatro unidades de logaritmo -unas "
        "cincuenta veces- que separan al residencial del industrial.",
    )
    g.figure.tight_layout(rect=[0, 0, 1, top])
    g.figure.savefig(ruta)
    plt.close(g.figure)


# -----------------------------------------------------------------------------
def main():
    if not CLUSTERS_PATH.exists():
        raise SystemExit(
            "Falta data/processed/clientes_con_cluster.csv. "
            "Ejecuta antes: python utils/codes/python/pca_clustering.py"
        )
    df = pd.read_csv(CLUSTERS_PATH)
    print(f"Datos: {len(df)} clientes con cluster asignado "
          f"(k = {df['cluster'].nunique()})\n")

    figuras = [
        ("01_heatmap_correlacion.png", heatmap_correlacion,
         "Matriz de correlacion anotada"),
        ("02_matriz_dispersion.png", matriz_dispersion,
         "Matriz de dispersion por cluster (PairGrid)"),
        ("03_clustermap.png", clustermap,
         "Clustermap con dendrogramas marginales"),
        ("04_perfil_clusters.png", perfil_clusters,
         "Perfil de clusteres en puntuaciones z"),
        ("05_distribuciones_cluster.png", distribuciones_cluster,
         "Violines de las variables clave por cluster"),
        ("06_regresion_temperatura.png", regresion_temperatura,
         "Regresion de consumo sobre temperatura por sector"),
    ]

    for nombre, funcion, descripcion in figuras:
        funcion(df, FIGURAS_DIR / nombre)
        print(f"  OK  {nombre:34s} {descripcion}")

    print(f"\nOK - Fase 3: {len(figuras)} figuras en "
          f"{FIGURAS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
