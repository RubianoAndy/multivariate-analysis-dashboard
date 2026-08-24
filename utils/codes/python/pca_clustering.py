"""Actividad 6 - Fase 2: analisis de componentes principales y clustering.

Nucleo analitico de la actividad. Se ejecuta en cuatro pasos:

1. **Preparacion.** El conjunto tiene tres variables numericas, una por
   concepto: cuanto consume el cliente (``consumo_kwh``), con que calidad
   electrica (``factor_potencia``) y desde hace cuanto (``antiguedad_anios``).

   ``consumo_kwh`` se transforma con logaritmo porque su distribucion es
   multiplicativa y fuertemente asimetrica; sin esa transformacion el analisis
   quedaria dominado por los pocos clientes industriales grandes. Despues se
   estandarizan las tres: el PCA sobre matriz de covarianzas daria todo el peso
   a la variable con unidades mas grandes, asi que se trabaja sobre la matriz de
   correlaciones (media 0, desviacion 1).

2. **PCA.** Se extraen las tres componentes, se reporta la varianza explicada y
   la acumulada, y se calculan las cargas como ``componente x sqrt(autovalor)``,
   que son las correlaciones entre cada variable original y cada componente.

   Con tan pocas variables **el criterio de Kaiser deja de servir**. Sobre
   matriz de correlaciones el autovalor medio vale 1 por construccion, de modo
   que el umbral "autovalor > 1" equivale a "por encima del promedio": con tres
   variables, la segunda componente se queda en 0.98 y quedaria descartada pese
   a recoger un tercio de la informacion. Se usa el criterio de **varianza
   acumulada del 80 %**, que retiene dos componentes (98.2 %), y el de Kaiser se
   sigue reportando en la tabla para dejar visible la discrepancia.

3. **Seleccion de k.** El clustering se hace sobre las componentes retenidas
   **estandarizadas**. Las puntuaciones crudas heredan la varianza del
   autovalor (1.97 en PC1 frente a 0.98 en PC2), asi que la distancia euclidea
   estaria dominada por la primera componente y K-Means acabaria partiendo a los
   clientes solo por el estado de su red, ignorando cuanto consumen. Igualar la
   escala de las componentes retenidas hace que ambas pesen lo mismo.

   Sobre ese espacio se recorre k = 2..8 evaluando inercia (metodo del codo),
   coeficiente de silueta, indice de Calinski-Harabasz y de Davies-Bouldin. El
   codo es ambiguo por construccion, asi que la decision se toma con la silueta
   y se contrasta con los otros dos indices.

4. **Clustering.** K-Means con el k elegido y, de forma independiente,
   aglomerativo de Ward. Se comparan con tabla de contingencia y con el indice
   Rand ajustado: si dos algoritmos con logicas distintas llegan a la misma
   particion, la estructura es de los datos y no del metodo.

Ejecucion (desde la raiz del proyecto):
    python utils/codes/python/pca_clustering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

from dataset import VARIABLES_NUMERICAS
from estilo import (
    aplicar_estilo_matplotlib,
    AZUL_UNISALLE, DORADO_UNISALLE, AZUL, ROJO, TEXTO, TEXTO_SUAVE,
    COLOR_SECTOR, ORDEN_SECTOR, COLOR_CLUSTER, ETIQUETAS_CORTAS,
    nombrar_componente,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "consumo_energia.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURAS_DIR = PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python" / "multivariate"
for d in (PROCESSED_DIR, FIGURAS_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
K_RANGO = range(2, 9)

# Porcentaje de varianza acumulada que deben alcanzar las componentes retenidas.
UMBRAL_VARIANZA = 80.0

# Variable cuyo proceso generador es multiplicativo: se analiza en logaritmo.
VARIABLES_LOG = ["consumo_kwh"]

# El modelo usa las tres variables numericas del conjunto.
VARIABLES_MODELO = list(VARIABLES_NUMERICAS)

aplicar_estilo_matplotlib()


def eje_componente(cargas, varianza, componente):
    """Rotulo de un eje: nombre de la componente, su concepto y su varianza.

    El concepto se deduce de la carga dominante en vez de escribirse a mano,
    porque el orden de las componentes cambia segun las variables que entren en
    el modelo -y en el dashboard, segun el filtro activo-.
    """
    fila = varianza.index[varianza["componente"] == componente][0]
    pct = varianza.loc[fila, "varianza_explicada_pct"]
    return f"{componente} - {nombrar_componente(cargas, componente)} ({pct:.1f} %)"


# -----------------------------------------------------------------------------
# 1. PREPARACION DE LA MATRIZ
# -----------------------------------------------------------------------------
def preparar_matriz(df):
    """Devuelve la matriz estandarizada lista para el PCA y el escalador usado.

    Retorna
    -------
    X : ndarray (n, 3)
        Matriz estandarizada.
    nombres : list[str]
        Nombre de cada columna, con el prefijo ``log_`` donde aplique.
    escalador : StandardScaler
        Necesario para devolver los centroides a unidades interpretables.
    """
    datos = df[VARIABLES_MODELO].copy()
    # VARIABLES_LOG lista todas las columnas de escala del conjunto, no solo las
    # del modelo: hay que intersecar para no buscar una que ya se descarto.
    for col in [c for c in VARIABLES_LOG if c in VARIABLES_MODELO]:
        datos[col] = np.log(datos[col])

    nombres = [f"log_{c}" if c in VARIABLES_LOG else c for c in VARIABLES_MODELO]
    escalador = StandardScaler()
    X = escalador.fit_transform(datos.values)
    return X, nombres, escalador


# -----------------------------------------------------------------------------
# 2. ANALISIS DE COMPONENTES PRINCIPALES
# -----------------------------------------------------------------------------
def ejecutar_pca(X, nombres):
    """Ajusta el PCA completo y devuelve varianza, cargas y puntuaciones."""
    pca = PCA(n_components=X.shape[1], random_state=SEED)
    scores = pca.fit_transform(X)

    autovalores = pca.explained_variance_
    varianza = pd.DataFrame(
        {
            "componente": [f"PC{i}" for i in range(1, len(autovalores) + 1)],
            "autovalor": autovalores.round(4),
            "varianza_explicada_pct": (pca.explained_variance_ratio_ * 100).round(2),
            "varianza_acumulada_pct": (
                np.cumsum(pca.explained_variance_ratio_) * 100
            ).round(2),
            "criterio_kaiser": np.where(autovalores > 1, "Retener", "Descartar"),
        }
    )
    # Criterio efectivo: la primera componente que alcanza el umbral acumulado.
    # Con tres variables Kaiser es demasiado exigente (ver punto 3 del encabezado).
    n_retenidas = int(
        (varianza["varianza_acumulada_pct"] < UMBRAL_VARIANZA).sum() + 1
    )
    n_retenidas = min(max(n_retenidas, 2), len(autovalores))
    varianza["criterio_varianza_acumulada"] = [
        "Retener" if i < n_retenidas else "Descartar" for i in range(len(autovalores))
    ]

    # Cargas = correlacion variable-componente. Se prefieren a los autovectores
    # crudos porque estan en [-1, 1] y son directamente interpretables.
    cargas = pd.DataFrame(
        pca.components_.T * np.sqrt(autovalores),
        index=nombres,
        columns=[f"PC{i}" for i in range(1, len(autovalores) + 1)],
    ).round(4)

    return pca, scores, varianza, cargas


def figura_scree(varianza, ruta):
    """Grafico de sedimentacion: varianza por componente y acumulada."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(varianza))

    barras = ax.bar(
        x, varianza["varianza_explicada_pct"], color=AZUL, width=0.62,
        label="Varianza explicada por componente",
    )
    for barra, valor in zip(barras, varianza["varianza_explicada_pct"]):
        if valor >= 1.5:
            ax.text(
                barra.get_x() + barra.get_width() / 2, valor + 1.2, f"{valor:.1f}%",
                ha="center", fontsize=8, color=TEXTO,
            )

    ax2 = ax.twinx()
    ax2.plot(
        x, varianza["varianza_acumulada_pct"], marker="o", color=AZUL_UNISALLE,
        linewidth=2, markersize=5, label="Varianza acumulada",
    )
    ax2.axhline(80, color=ROJO, linestyle="--", linewidth=1.2)
    ax2.text(len(x) - 0.4, 81.5, "80 %", color=ROJO, fontsize=8, ha="right")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Varianza acumulada (%)", color=TEXTO)
    ax2.grid(False)

    retenidas = int((varianza["criterio_kaiser"] == "Retener").sum())
    ax.axvline(retenidas - 0.5, color=DORADO_UNISALLE, linewidth=2.5, alpha=0.9)
    ax.text(
        retenidas - 0.42, ax.get_ylim()[1] * 0.62,
        f" Criterio de Kaiser:\n {retenidas} componentes (autovalor > 1)",
        fontsize=8, color=TEXTO, va="top",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(varianza["componente"])
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_title(
        "Grafico de sedimentacion (scree plot)\n"
        f"Las primeras {retenidas} componentes resumen "
        f"{varianza.loc[retenidas - 1, 'varianza_acumulada_pct']:.1f} % de la informacion",
        loc="left",
    )

    lineas, etiquetas = ax.get_legend_handles_labels()
    l2, e2 = ax2.get_legend_handles_labels()
    ax.legend(lineas + l2, etiquetas + e2, loc="center right", fontsize=9)

    fig.savefig(ruta)
    plt.close(fig)


def figura_biplot(scores, cargas, df, ruta, varianza):
    """Biplot PC1-PC2: observaciones por sector y vectores de carga."""
    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    for sector in ORDEN_SECTOR:
        m = (df["sector"] == sector).values
        ax.scatter(
            scores[m, 0], scores[m, 1], s=34, alpha=0.72,
            color=COLOR_SECTOR[sector], edgecolor="white", linewidth=0.5, label=sector,
        )

    # Los vectores se escalan al rango de las puntuaciones para que sean visibles
    # sobre la nube; el factor no altera la interpretacion (direccion y longitud
    # relativa son lo que importa).
    escala = 0.82 * max(np.abs(scores[:, :2]).max(), 1)

    # Seis de las nueve variables apuntan casi en la misma direccion (todas miden
    # tamano), asi que sus etiquetas se apilarian unas sobre otras. Alejarlas del
    # origen no sirve -seguirian colineales-, hay que escalonarlas en
    # perpendicular a su propio vector. Para cada flecha se prueban desplazamientos
    # perpendiculares crecientes (0, +1, -1, +2, -2 ...) y se toma el primero que
    # deja libre el hueco respecto a las etiquetas ya situadas.
    colocadas = []
    separacion = 0.26 * escala
    orden = cargas.reindex(
        cargas["PC1"].abs().add(cargas["PC2"].abs()).sort_values(ascending=False).index
    )

    for variable in orden.index:
        x, y = cargas.loc[variable, "PC1"], cargas.loc[variable, "PC2"]
        ax.arrow(
            0, 0, x * escala, y * escala, color=AZUL_UNISALLE, alpha=0.85,
            width=0.005 * escala, head_width=0.05 * escala, length_includes_head=True,
        )

        # Vector unitario perpendicular a la flecha.
        norma = np.hypot(x, y) or 1.0
        perp = (-y / norma, x / norma)
        base = (x * escala * 1.09, y * escala * 1.09)

        etiqueta = ETIQUETAS_CORTAS.get(variable.replace("log_", ""), variable)
        destino = base
        for paso in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
            candidato = (
                base[0] + perp[0] * separacion * paso,
                base[1] + perp[1] * separacion * paso,
            )
            if all(
                np.hypot(candidato[0] - qx, candidato[1] - qy) >= separacion
                for qx, qy in colocadas
            ):
                destino = candidato
                break
        colocadas.append(destino)

        ax.annotate(
            etiqueta, xy=(x * escala, y * escala), xytext=destino,
            color=AZUL_UNISALLE, fontsize=8.5, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color=TEXTO_SUAVE, linewidth=0.6,
                            shrinkA=0, shrinkB=2, alpha=0.7),
        )

    ax.axhline(0, color=TEXTO_SUAVE, linewidth=0.8)
    ax.axvline(0, color=TEXTO_SUAVE, linewidth=0.8)

    # Encuadre: la nube y todas las etiquetas, con holgura extra arriba para que
    # el titulo no se monte sobre la etiqueta mas alta.
    xs = [p[0] for p in colocadas] + list(scores[:, 0])
    ys = [p[1] for p in colocadas] + list(scores[:, 1])
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 1.1)

    ax.set_xlabel(eje_componente(cargas, varianza, "PC1"))
    ax.set_ylabel(eje_componente(cargas, varianza, "PC2"))
    ax.set_title(
        "Biplot: clientes y variables en el plano de las dos primeras componentes",
        loc="left",
    )
    ax.legend(title="Sector", loc="lower right", frameon=True, framealpha=0.9)

    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. SELECCION DEL NUMERO DE GRUPOS
# -----------------------------------------------------------------------------
def evaluar_k(Z, rango=K_RANGO):
    """Recorre valores de k y calcula cuatro indices de validacion interna."""
    filas = []
    for k in rango:
        km = KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(Z)
        etiquetas = km.labels_
        filas.append(
            {
                "k": k,
                "inercia": round(km.inertia_, 2),
                "silueta": round(silhouette_score(Z, etiquetas), 4),
                "calinski_harabasz": round(calinski_harabasz_score(Z, etiquetas), 2),
                "davies_bouldin": round(davies_bouldin_score(Z, etiquetas), 4),
            }
        )
    tabla = pd.DataFrame(filas)
    tabla["mejor_por_silueta"] = tabla["silueta"] == tabla["silueta"].max()
    return tabla


def figura_seleccion_k(tabla, k_elegido, ruta):
    """Panel doble: metodo del codo y coeficiente de silueta."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(tabla["k"], tabla["inercia"], marker="o", color=AZUL, linewidth=2)
    ax1.axvline(k_elegido, color=DORADO_UNISALLE, linewidth=2.5, alpha=0.9)
    ax1.set_xlabel("Numero de clusteres (k)")
    ax1.set_ylabel("Inercia (suma de cuadrados intra-cluster)")
    ax1.set_title("Metodo del codo", loc="left")

    ax2.plot(tabla["k"], tabla["silueta"], marker="o", color=AZUL_UNISALLE, linewidth=2)
    mejor = tabla.loc[tabla["silueta"].idxmax()]
    ax2.scatter([mejor["k"]], [mejor["silueta"]], s=140, facecolor="none",
                edgecolor=ROJO, linewidth=2, zorder=5)
    ax2.annotate(
        f"maximo: k = {int(mejor['k'])}\nsilueta = {mejor['silueta']:.3f}",
        xy=(mejor["k"], mejor["silueta"]), xytext=(10, -28),
        textcoords="offset points", fontsize=9, color=TEXTO,
    )
    ax2.set_xlabel("Numero de clusteres (k)")
    ax2.set_ylabel("Coeficiente de silueta medio")
    ax2.set_title("Criterio de la silueta", loc="left")

    fig.suptitle(
        "Seleccion del numero de grupos: el codo sugiere, la silueta decide",
        fontsize=12, fontweight="bold", color=AZUL_UNISALLE, x=0.008, ha="left",
    )
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_silueta(Z, etiquetas, ruta):
    """Diagrama de silueta por observacion, agrupado por cluster."""
    valores = silhouette_samples(Z, etiquetas)
    media = valores.mean()
    k = len(np.unique(etiquetas))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    y_inferior = 5
    for c in range(k):
        v = np.sort(valores[etiquetas == c])
        y_superior = y_inferior + len(v)
        ax.fill_betweenx(
            np.arange(y_inferior, y_superior), 0, v,
            facecolor=COLOR_CLUSTER[c % len(COLOR_CLUSTER)], alpha=0.85, linewidth=0,
        )
        ax.text(-0.045, y_inferior + len(v) / 2, f"C{c}", va="center",
                fontsize=9, fontweight="bold", color=TEXTO)
        y_inferior = y_superior + 5

    ax.axvline(media, color=ROJO, linestyle="--", linewidth=1.4)
    ax.text(media + 0.01, y_inferior * 0.97, f"media = {media:.3f}",
            color=ROJO, fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_ylabel("Clientes ordenados dentro de cada cluster")
    ax.set_title(
        f"Diagrama de silueta con k = {k}\n"
        "Valores negativos senalan clientes mejor ubicados en otro grupo",
        loc="left",
    )
    ax.grid(axis="y", visible=False)
    fig.savefig(ruta)
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. CLUSTERING Y PERFILES
# -----------------------------------------------------------------------------
def figura_dendrograma(Z, k, ruta):
    """Dendrograma de Ward, cortado a la altura que produce k grupos."""
    enlace = linkage(Z, method="ward")
    # Altura de corte: punto medio entre las dos fusiones que definen k grupos.
    alturas = enlace[:, 2]
    corte = (alturas[-k] + alturas[-k + 1]) / 2 if k > 1 else alturas[-1]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    from scipy.cluster import hierarchy

    hierarchy.set_link_color_palette(
        [COLOR_CLUSTER[i % len(COLOR_CLUSTER)] for i in range(k)]
    )
    dendrogram(
        enlace, ax=ax, color_threshold=corte, above_threshold_color=TEXTO_SUAVE,
        no_labels=True,
    )
    ax.axhline(corte, color=ROJO, linestyle="--", linewidth=1.4)
    ax.text(
        ax.get_xlim()[1] * 0.995, corte * 1.03, f" corte en k = {k}",
        color=ROJO, fontsize=9, ha="right",
    )
    ax.set_xlabel("Clientes (300 hojas)")
    ax.set_ylabel("Distancia de fusion (Ward)")
    ax.set_title(
        "Dendrograma jerarquico aglomerativo (enlace de Ward)\n"
        "Alternativa a K-Means: no exige fijar k de antemano",
        loc="left",
    )
    ax.grid(axis="x", visible=False)
    hierarchy.set_link_color_palette(None)
    fig.savefig(ruta)
    plt.close(fig)
    return enlace, corte


def figura_clusters(scores, etiquetas, centroides_pca, ruta, cargas, varianza):
    """Plano PC1-PC2 coloreado por cluster, con centroides marcados."""
    fig, ax = plt.subplots(figsize=(9, 6.8))
    k = len(np.unique(etiquetas))

    for c in range(k):
        m = etiquetas == c
        ax.scatter(
            scores[m, 0], scores[m, 1], s=36, alpha=0.75,
            color=COLOR_CLUSTER[c % len(COLOR_CLUSTER)],
            edgecolor="white", linewidth=0.5, label=f"Cluster {c} (n={m.sum()})",
        )
    ax.scatter(
        centroides_pca[:, 0], centroides_pca[:, 1], marker="X", s=230,
        color=AZUL_UNISALLE, edgecolor="white", linewidth=1.6, zorder=6,
        label="Centroides",
    )

    ax.axhline(0, color=TEXTO_SUAVE, linewidth=0.8)
    ax.axvline(0, color=TEXTO_SUAVE, linewidth=0.8)
    ax.set_xlabel(eje_componente(cargas, varianza, "PC1"))
    ax.set_ylabel(eje_componente(cargas, varianza, "PC2"))
    ax.set_title(
        f"Particion de K-Means con k = {k} proyectada sobre el plano principal",
        loc="left",
    )
    ax.legend(loc="best", fontsize=9)
    fig.savefig(ruta)
    plt.close(fig)


def perfilar(df, etiquetas, nombres_pca, scores_ret):
    """Construye las tablas de perfil de cada cluster.

    Retorna
    -------
    perfil : DataFrame
        Media de cada variable en unidades originales, por cluster.
    perfil_z : DataFrame
        Las mismas medias en puntuaciones z respecto a la media global; es la
        tabla que se interpreta, porque hace comparables variables con unidades
        distintas.

        Las z se calculan sobre la **matriz transformada**, la misma que usa el
        PCA. Con el consumo en su escala original la cola industrial infla la
        desviacion tipica y comprime a todos los demas: un grupo de 3 500 kWh
        -veinte veces la media residencial- se quedaria en z = 0.46 y pareceria
        moderado. En logaritmo, que es donde el consumo se distribuye de forma
        simetrica, la z dice lo que se espera que diga.
    composicion : DataFrame
        Reparto de sectores y regiones dentro de cada cluster.
    """
    datos = df.copy()
    datos["cluster"] = etiquetas

    perfil = datos.groupby("cluster")[VARIABLES_NUMERICAS].mean().round(2)
    perfil.insert(0, "n_clientes", datos.groupby("cluster").size())

    transformada = df[VARIABLES_NUMERICAS].copy()
    for col in [c for c in VARIABLES_LOG if c in transformada.columns]:
        transformada[col] = np.log(transformada[col])
    transformada["cluster"] = etiquetas

    media = transformada[VARIABLES_NUMERICAS].mean()
    desv = transformada[VARIABLES_NUMERICAS].std()
    perfil_z = (
        (transformada.groupby("cluster")[VARIABLES_NUMERICAS].mean() - media) / desv
    ).round(3)

    # Media de las componentes retenidas: resume el cluster en el espacio del PCA.
    comp = pd.DataFrame(scores_ret, columns=nombres_pca)
    comp["cluster"] = etiquetas
    perfil_pca = comp.groupby("cluster").mean().round(3)

    sector = pd.crosstab(datos["cluster"], datos["sector"])
    region = pd.crosstab(datos["cluster"], datos["region"])
    composicion = sector.join(region, lsuffix="_sector", rsuffix="_region")

    return perfil, perfil_z, perfil_pca, composicion


def describir(valor, todos, umbral_alto, umbral_bajo, palabras):
    """Traduce un indice a una palabra, combinando escala absoluta y relativa.

    Fuera de la banda neutra manda el umbral absoluto: una z de +0.8 es alta
    diga lo que diga el resto de grupos. Dentro de ella el umbral no distingue
    nada, y ahi se recurre a la posicion respecto a la mediana de los grupos de
    esta misma ejecucion. Sin ese desempate, al analizar un subconjunto
    homogeneo -un solo sector, por ejemplo- todos los grupos caen en la banda
    neutra y acaban recibiendo la misma descripcion.

    Parametros
    ----------
    valor : float
        Indice del cluster que se esta describiendo.
    todos : array-like
        Indices de todos los clusteres de la ejecucion.
    umbral_alto, umbral_bajo : float
        Limites de la banda neutra.
    palabras : tuple[str, str, str, str]
        Descriptores para alto, medio-alto, medio-bajo y bajo.
    """
    if valor > umbral_alto:
        return palabras[0]
    if valor < umbral_bajo:
        return palabras[3]
    return palabras[1] if valor >= np.median(todos) else palabras[2]


def nombrar_clusters(perfil_z):
    """Asigna una etiqueta legible a cada cluster segun su perfil en z.

    La regla es deliberadamente simple y explicita: el tamano lo da la z del
    consumo y el estado de la red, el promedio de la z del factor de potencia
    contra la de la antiguedad -que van en sentidos opuestos-. El nombre no
    interviene en el modelo, solo en la comunicacion de resultados.
    """
    tamano = perfil_z["consumo_kwh"]
    calidad = (perfil_z["factor_potencia"] - perfil_z["antiguedad_anios"]) / 2

    nombres = {}
    for c in perfil_z.index:
        t = describir(
            tamano[c], tamano.values, 0.5, -0.3,
            ("gran consumidor", "consumidor medio-alto",
             "consumidor medio-bajo", "consumidor pequeno"),
        )
        q = describir(
            calidad[c], calidad.values, 0.25, -0.25,
            ("red confiable", "red aceptable", "red desgastada", "red degradada"),
        )
        nombres[c] = f"C{c}: {t}, {q}"

    return pd.DataFrame(
        {
            "cluster": list(nombres.keys()),
            "etiqueta": list(nombres.values()),
            "indice_tamano": tamano.round(3).values,
            "indice_calidad": calidad.round(3).values,
        }
    )


# -----------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATASET_PATH)

    X, nombres, escalador = preparar_matriz(df)
    print(f"Matriz estandarizada: {X.shape[0]} clientes x {X.shape[1]} variables\n")

    # --- PCA -----------------------------------------------------------------
    pca, scores, varianza, cargas = ejecutar_pca(X, nombres)
    varianza.to_csv(PROCESSED_DIR / "pca_varianza.csv", index=False)
    cargas.to_csv(PROCESSED_DIR / "pca_cargas.csv")

    n_ret = int((varianza["criterio_varianza_acumulada"] == "Retener").sum())
    nombres_pca = [f"PC{i}" for i in range(1, n_ret + 1)]
    scores_ret = scores[:, :n_ret]

    # Espacio de clustering: componentes retenidas con varianza igualada, para
    # que PC1 no monopolice la distancia euclidea (ver encabezado, punto 3).
    sigma_pc = scores_ret.std(axis=0)
    Z = scores_ret / sigma_pc

    print("1. VARIANZA EXPLICADA")
    print(varianza.to_string(index=False))
    print(
        f"\nComponentes retenidas (varianza acumulada > {UMBRAL_VARIANZA:.0f} %): {n_ret} -> "
        f"{varianza.loc[n_ret - 1, 'varianza_acumulada_pct']:.2f} % de la varianza total\n"
    )
    print("2. CARGAS DE LAS COMPONENTES RETENIDAS")
    print(cargas[nombres_pca].to_string(), "\n")

    figura_scree(varianza, FIGURAS_DIR / "01_scree_varianza.png")
    figura_biplot(scores, cargas, df, FIGURAS_DIR / "02_biplot_pca.png", varianza)

    # --- Seleccion de k ------------------------------------------------------
    tabla_k = evaluar_k(Z)
    tabla_k.to_csv(PROCESSED_DIR / "seleccion_k.csv", index=False)
    k = int(tabla_k.loc[tabla_k["silueta"].idxmax(), "k"])
    print("3. SELECCION DEL NUMERO DE GRUPOS")
    print(tabla_k.to_string(index=False))
    print(f"\nk elegido por maxima silueta: {k}\n")

    figura_seleccion_k(tabla_k, k, FIGURAS_DIR / "03_seleccion_k.png")

    # --- K-Means -------------------------------------------------------------
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(Z)
    etiquetas = km.labels_
    sil = silhouette_score(Z, etiquetas)

    # Los centroides viven en el espacio estandarizado; se deshace primero esa
    # estandarizacion, luego el PCA y por ultimo el logaritmo, para leerlos en
    # kWh, m2 y anios en vez de en desviaciones tipicas.
    centroides_pc = km.cluster_centers_ * sigma_pc
    centroides_std = pca.inverse_transform(
        np.hstack([centroides_pc, np.zeros((k, X.shape[1] - n_ret))])
    )
    centroides = pd.DataFrame(
        escalador.inverse_transform(centroides_std), columns=nombres
    )
    for col in [c for c in centroides.columns if c.startswith("log_")]:
        centroides[col] = np.exp(centroides[col])
    centroides.columns = VARIABLES_MODELO
    centroides.index.name = "cluster"
    centroides.round(2).to_csv(PROCESSED_DIR / "centroides_kmeans.csv")

    figura_silueta(Z, etiquetas, FIGURAS_DIR / "04_silueta_clusters.png")
    figura_clusters(scores, etiquetas, centroides_pc,
                    FIGURAS_DIR / "05_clusters_pca.png", cargas, varianza)

    # --- Jerarquico y comparacion -------------------------------------------
    enlace, corte = figura_dendrograma(Z, k, FIGURAS_DIR / "06_dendrograma.png")
    jerarquico = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    ari = adjusted_rand_score(etiquetas, jerarquico)
    contingencia = pd.crosstab(
        pd.Series(etiquetas, name="kmeans"),
        pd.Series(jerarquico, name="jerarquico_ward"),
    )
    contingencia.to_csv(PROCESSED_DIR / "kmeans_vs_jerarquico.csv")

    print("4. K-MEANS FRENTE A JERARQUICO DE WARD")
    print(contingencia.to_string())
    print(f"Indice Rand ajustado (ARI): {ari:.4f}")
    print(f"Silueta media (K-Means, k={k}): {sil:.4f}\n")

    # --- Perfiles ------------------------------------------------------------
    perfil, perfil_z, perfil_pca, composicion = perfilar(df, etiquetas, nombres_pca, scores_ret)
    etiquetas_cluster = nombrar_clusters(perfil_z)

    perfil.to_csv(PROCESSED_DIR / "perfiles_cluster.csv")
    perfil_z.to_csv(PROCESSED_DIR / "perfiles_cluster_z.csv")
    perfil_pca.to_csv(PROCESSED_DIR / "perfiles_cluster_componentes.csv")
    composicion.to_csv(PROCESSED_DIR / "composicion_cluster.csv")
    etiquetas_cluster.to_csv(PROCESSED_DIR / "etiquetas_cluster.csv", index=False)

    print("5. PERFIL DE LOS CLUSTERES (medias en unidades originales)")
    print(perfil.to_string())
    print("\n   Perfil en puntuaciones z (respecto a la media global)")
    print(perfil_z.to_string())
    print("\n   Composicion por sector y region")
    print(composicion.to_string())
    print("\n   Interpretacion")
    print(etiquetas_cluster.to_string(index=False), "\n")

    # --- Dataset enriquecido -------------------------------------------------
    salida = df.copy()
    for i, nombre in enumerate(nombres_pca):
        salida[nombre] = scores_ret[:, i].round(4)
    salida["cluster"] = etiquetas
    salida["cluster_jerarquico"] = jerarquico
    salida["silueta"] = silhouette_samples(Z, etiquetas).round(4)
    salida.to_csv(PROCESSED_DIR / "clientes_con_cluster.csv", index=False)

    # --- Resumen del modelo --------------------------------------------------
    resumen = pd.DataFrame(
        [
            {"metrica": "n_clientes", "valor": len(df)},
            {"metrica": "n_variables_modelo", "valor": len(VARIABLES_MODELO)},
            {"metrica": "componentes_retenidas", "valor": n_ret},
            {"metrica": "variables_del_modelo", "valor": " | ".join(VARIABLES_MODELO)},
            {"metrica": "varianza_acumulada_pct",
             "valor": float(varianza.loc[n_ret - 1, "varianza_acumulada_pct"])},
            {"metrica": "k_optimo", "valor": k},
            {"metrica": "silueta_media", "valor": round(sil, 4)},
            {"metrica": "calinski_harabasz",
             "valor": round(calinski_harabasz_score(Z, etiquetas), 2)},
            {"metrica": "davies_bouldin",
             "valor": round(davies_bouldin_score(Z, etiquetas), 4)},
            {"metrica": "ari_kmeans_vs_ward", "valor": round(ari, 4)},
        ]
    )
    resumen.to_csv(PROCESSED_DIR / "resumen_modelo.csv", index=False)

    print("OK - Fase 2: 11 tablas en data/processed/ y 6 figuras en")
    print(f"   {FIGURAS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
