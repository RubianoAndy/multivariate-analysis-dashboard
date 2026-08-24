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
from scipy.cluster.hierarchy import linkage
from statsmodels.formula.api import ols

from dataset import VARIABLES_NUMERICAS
from pca_clustering import VARIABLES_LOG, VARIABLES_MODELO
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

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
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
        "Matriz de correlacion entre las tres variables del modelo",
        "El factor de potencia y la antiguedad estan correlacionadas a -0.94: describen la "
        "misma realidad. Esa redundancia es justo lo que el PCA comprime en una componente.",
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 2. MATRIZ DE DISPERSION POR CLUSTER
# -----------------------------------------------------------------------------
def matriz_dispersion(df, ruta):
    """PairGrid de las tres variables del modelo, coloreado por cluster.

    Con tres variables la matriz cabe entera -seis paneles- y se puede leer
    panel por panel, que es justamente lo que se perdia con nueve.
    """
    datos = df[VARIABLES_MODELO + ["cluster"]].copy()
    for col in [c for c in VARIABLES_LOG if c in VARIABLES_MODELO]:
        datos[col] = np.log(datos[col])
    datos = datos.rename(
        columns={
            c: (f"log {ETIQUETAS_CORTAS[c]}" if c in VARIABLES_LOG else ETIQUETAS_CORTAS[c])
            for c in VARIABLES_MODELO
        }
    )
    datos["Cluster"] = datos["cluster"].map(lambda c: f"C{c}")
    datos = datos.drop(columns="cluster")

    paleta = {f"C{c}": COLOR_CLUSTER[c % len(COLOR_CLUSTER)]
              for c in sorted(df["cluster"].unique())}

    g = sns.PairGrid(datos, hue="Cluster", hue_order=sorted(paleta), palette=paleta,
                     diag_sharey=False, height=2.15, corner=True)
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
        "Matriz de dispersion de las tres variables del modelo, por cluster",
        "Los cuatro grupos se separan en el par que cruza consumo con estado de la red; "
        "dentro del par factor de potencia - antiguedad se alinean sobre una recta, que es "
        "la redundancia que el PCA comprime en una sola componente.",
    )
    g.figure.subplots_adjust(top=top)
    g.figure.savefig(ruta)
    plt.close(g.figure)


# -----------------------------------------------------------------------------
# 3. CLUSTERMAP
# -----------------------------------------------------------------------------
def clustermap(df, ruta):
    """Mapa de calor de los datos estandarizados con dendrogramas en los margenes.

    Es la unica figura que muestra el resultado **cliente por cliente** en vez
    de por promedios: las demas resumen cada grupo en una fila, y aqui se ve que
    los bloques son homogeneos individuo a individuo y no un efecto de promediar.

    Con tres variables el dendrograma de columnas es trivial -tiene tres hojas- y
    no aporta nada; el valor esta en el de filas, que reordena a los 300 clientes
    por similitud sin conocer la particion y deberia reproducir los mismos
    bloques que K-Means marca en la franja de color de la izquierda.
    """
    datos = df[VARIABLES_MODELO].copy()
    for col in [c for c in VARIABLES_LOG if c in VARIABLES_MODELO]:
        datos[col] = np.log(datos[col])
    datos = (datos - datos.mean()) / datos.std()
    enlace_filas = linkage(datos.values, method="ward")

    matriz = pd.DataFrame(
        datos.values,
        columns=[
            f"log {ETIQUETAS_CORTAS[c]}" if c in VARIABLES_LOG else ETIQUETAS_CORTAS[c]
            for c in VARIABLES_MODELO
        ],
    )

    colores_fila = df["cluster"].map(
        lambda c: COLOR_CLUSTER[c % len(COLOR_CLUSTER)]
    ).rename("Cluster")

    g = sns.clustermap(
        matriz,
        method="ward",
        row_linkage=enlace_filas,
        cmap=ESCALA_DIVERGENTE,
        center=0,
        vmin=-2.5,
        vmax=2.5,
        row_colors=colores_fila.values,
        yticklabels=False,
        figsize=(7.4, 7.2),
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
        "El dendrograma de la izquierda reordena a los 300 clientes por similitud sin conocer "
        "la particion, y forma los mismos bloques que K-Means marca en la franja de color: la "
        "estructura se sostiene cliente a cliente, no solo en los promedios.",
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

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
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
        "La tabla de interpretacion hecha figura: cada celda dice cuantas desviaciones tipicas "
        "se aparta el grupo de la media global. De aqui salen los nombres de los segmentos.",
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
    ]
    orden = sorted(df["cluster"].unique())
    paleta = [COLOR_CLUSTER[c % len(COLOR_CLUSTER)] for c in orden]
    etiquetas_x = [f"C{c}" for c in orden]

    fig, ejes = plt.subplots(1, 3, figsize=(10.5, 4.3))
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
        "Distribucion de las tres variables dentro de cada cluster",
        "El violin muestra la forma completa de la distribucion, no solo los cuartiles: deja "
        "ver si un grupo es compacto de verdad o si su promedio esconde dos modas.",
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. REGRESION: LA COLINEALIDAD QUE MOTIVA EL PCA
# -----------------------------------------------------------------------------
def regresion_colinealidad(df, ruta):
    """Version grafica de la regresion de la Fase 1, en dos paneles.

    El panel izquierdo ensena el problema: el factor de potencia y la antiguedad
    caen casi sobre una recta (r = -0.94). Miden lo mismo, y una regresion que
    los use como predictores separados no tiene forma de repartir el efecto
    entre ellos.

    El panel derecho ensena la consecuencia: la relacion entre el consumo y el
    factor de potencia dentro de cada sector, con la pendiente conjunta y su
    intervalo de confianza. La estimacion queda muy lejos del valor fisico que
    fija el generador (unos -1.1) por dos motivos que se suman: la colinealidad
    infla la varianza del coeficiente, y el modelo omite el tamano de la
    instalacion -que este conjunto no mide- y que empuja el consumo hacia
    arriba. El resultado es un coeficiente que no se puede interpretar.

    Es la motivacion directa del PCA de la Fase 2, que en vez de pedirle a dos
    variables colineales que se repartan un efecto las comprime en una sola
    componente.
    """
    datos = df.assign(log_consumo=np.log(df["consumo_kwh"]))
    r = datos["factor_potencia"].corr(datos["antiguedad_anios"])

    # Las cifras del recuadro se recalculan aqui en vez de escribirse a mano:
    # si cambian los datos, la anotacion cambia con ellos.
    modelo = ols(
        "log_consumo ~ factor_potencia + antiguedad_anios + C(sector)", data=datos
    ).fit()
    beta = modelo.params["factor_potencia"]
    ic = modelo.conf_int().loc["factor_potencia"]
    r2_entre_predictores = ols(
        "factor_potencia ~ antiguedad_anios", data=datos
    ).fit().rsquared
    vif = 1 / (1 - r2_entre_predictores)
    # El generador fija consumo proporcional a 0.92 / factor_potencia, de modo
    # que la pendiente fisica es -1 / factor_potencia medio.
    pendiente_real = -1 / datos["factor_potencia"].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- Panel izquierdo: los dos predictores son la misma variable ----------
    for sector in ORDEN_SECTOR:
        m = datos["sector"] == sector
        ax1.scatter(
            datos.loc[m, "factor_potencia"], datos.loc[m, "antiguedad_anios"],
            s=22, alpha=0.65, color=COLOR_SECTOR[sector], edgecolor="none",
            label=sector,
        )
    sns.regplot(
        data=datos, x="factor_potencia", y="antiguedad_anios", scatter=False,
        color=TEXTO, line_kws={"linewidth": 2, "linestyle": "--"}, ci=None, ax=ax1,
    )
    ax1.set_xlabel("Factor de potencia")
    ax1.set_ylabel("Antiguedad de la instalacion (anios)")
    ax1.set_title(f"Los dos predictores: r = {r:.2f}", loc="left", fontsize=11)
    ax1.legend(title="Sector", fontsize=8.5, title_fontsize=9, loc="upper right",
               frameon=True, framealpha=0.94, edgecolor=BORDE, facecolor="white")

    # --- Panel derecho: la pendiente que no se puede estimar bien ------------
    for sector in ORDEN_SECTOR:
        m = datos["sector"] == sector
        sns.regplot(
            data=datos[m], x="factor_potencia", y="log_consumo",
            scatter_kws={"s": 22, "alpha": 0.6, "edgecolor": "none"},
            line_kws={"linewidth": 2}, color=COLOR_SECTOR[sector], ci=95, ax=ax2,
        )
    ax2.set_xlabel("Factor de potencia")
    ax2.set_ylabel("Consumo mensual, ln(kWh)")
    ax2.set_title("El efecto sobre el consumo, por sector", loc="left", fontsize=11)
    ax2.annotate(
        "Regresion conjunta:\n"
        f"  pendiente estimada = {beta:.2f}\n"
        f"  IC 95 % = [{ic.iloc[0]:.2f}, {ic.iloc[1]:.2f}]\n"
        f"  valor fisico ~ {pendiente_real:.2f}\n"
        f"  VIF = {vif:.1f}",
        xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8.5, color=TEXTO,
        va="bottom", ha="left", linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor=BORDE, alpha=0.94),
    )

    top = titulo(
        fig,
        "Por que hacen falta las componentes principales",
        f"Factor de potencia y antiguedad describen la misma realidad (r = {r:.2f}), asi que "
        f"la regresion no puede repartir el efecto entre ambos: el VIF sube a {vif:.1f} y el "
        f"coeficiente se va a {beta:.1f}, lejos del valor fisico de {pendiente_real:.1f}. El PCA "
        "resuelve esto comprimiendo las dos variables en una componente.",
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(ruta)
    plt.close(fig)


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
        ("06_regresion_colinealidad.png", regresion_colinealidad,
         "Regresion con predictores colineales (motiva el PCA)"),
    ]

    for nombre, funcion, descripcion in figuras:
        funcion(df, FIGURAS_DIR / nombre)
        print(f"  OK  {nombre:34s} {descripcion}")

    print(f"\nOK - Fase 3: {len(figuras)} figuras en "
          f"{FIGURAS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
