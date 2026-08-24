"""Actividad 6 - Fase 4: visualizacion interactiva con Plotly.

Las figuras de Matplotlib y Seaborn son el soporte del informe escrito: una
imagen fija, con una lectura ya decidida por quien la produce. Plotly cambia el
contrato. Cada figura de esta fase se exporta como HTML autocontenido en el que
el lector puede hacer zoom, aislar una serie, rotar el espacio o leer los datos
de un cliente concreto al pasar el cursor: la interpretacion deja de estar
cerrada de antemano.

Se generan cuatro figuras, elegidas porque cada una aporta algo que la version
estatica no puede dar:

* **Biplot interactivo** - el mismo plano PC1-PC2 de la Fase 2, pero con el
  identificador y los valores originales de cada cliente en el tooltip: permite
  pasar del punto atipico al cliente que lo produce.
* **Dispersion 3D** - como el modelo usa exactamente tres variables, estos tres
  ejes son el espacio completo del analisis, sin proyeccion ni perdida: lo que
  se ve al rotar es el dato entero, y permite comprobar que la particion no es
  un artefacto de mirar solo dos dimensiones.
* **Coordenadas paralelas** - las nueve variables en un solo eje comun, con
  filtros de rango arrastrables sobre cada eje. Es la forma habitual de
  explorar perfiles multivariantes.
* **Sunburst** - la jerarquia cluster > sector > region, para ver de que esta
  compuesto cada segmento sin leer una tabla de contingencia de tres entradas.

Cada figura se guarda en HTML (interactiva, para el dashboard y la defensa) y en
PNG (estatica, para el informe en PDF). El PNG se produce con kaleido; si no
esta disponible se avisa y se continua, porque el HTML es el entregable real.

Ejecucion (desde la raiz del proyecto):
    python utils/codes/python/interactive_viz.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pca_clustering import (
    VARIABLES_LOG, VARIABLES_MODELO, preparar_matriz, ejecutar_pca, eje_componente,
)
from estilo import (
    PLANTILLA_PLOTLY, titulo_plotly, AZUL_UNISALLE, BORDE,
    COLOR_CLUSTER, ETIQUETAS_CORTAS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLUSTERS_PATH = PROCESSED_DIR / "clientes_con_cluster.csv"
FIGURAS_DIR = PROJECT_ROOT / "public" / "assets" / "images" / "figures" / "python" / "interactive"
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

# Ancho y alto de los PNG exportados; el HTML es responsivo y no los usa.
ANCHO_PNG, ALTO_PNG = 1200, 750


def color_cluster(c):
    return COLOR_CLUSTER[int(c) % len(COLOR_CLUSTER)]


def guardar(fig, nombre, exportar_png=True):
    """Escribe la figura como HTML interactivo y, si se puede, como PNG.

    ``include_plotlyjs="directory"`` deja una unica copia de ``plotly.min.js``
    en la carpeta y hace que los cuatro HTML la referencien. Incrustarla en cada
    archivo (``True``) los dejaria en 4.7 MB cada uno -19 MB para el conjunto-;
    enlazarla a un CDN los dejaria diminutos pero inservibles sin conexion.
    Compartir el archivo da lo mejor de ambas: se abren con doble clic, sin
    servidor ni red, mientras la carpeta viaje completa.
    """
    ruta_html = FIGURAS_DIR / f"{nombre}.html"
    fig.write_html(
        ruta_html,
        include_plotlyjs="directory",
        full_html=True,
        config={"displaylogo": False, "responsive": True},
    )
    estado = "HTML"

    if exportar_png:
        try:
            fig.write_image(FIGURAS_DIR / f"{nombre}.png",
                            width=ANCHO_PNG, height=ALTO_PNG, scale=2)
            estado = "HTML + PNG"
        except Exception as exc:  # kaleido ausente o sin navegador disponible
            estado = f"HTML (PNG omitido: {type(exc).__name__})"
    return estado


def etiquetas_cluster(df):
    """Diccionario cluster -> nombre legible, leido de la Fase 2 si existe."""
    ruta = PROCESSED_DIR / "etiquetas_cluster.csv"
    if ruta.exists():
        tabla = pd.read_csv(ruta)
        return dict(zip(tabla["cluster"], tabla["etiqueta"]))
    return {c: f"C{c}" for c in sorted(df["cluster"].unique())}


# -----------------------------------------------------------------------------
# 1. BIPLOT INTERACTIVO
# -----------------------------------------------------------------------------
def biplot_interactivo(df, cargas, varianza, nombres_cluster, ruta):
    """Plano PC1-PC2 con tooltip por cliente y vectores de carga superpuestos."""
    datos = df.copy()
    datos["Cluster"] = datos["cluster"].map(nombres_cluster)

    fig = px.scatter(
        datos,
        x="PC1", y="PC2",
        color="Cluster",
        color_discrete_map={
            nombres_cluster[c]: color_cluster(c) for c in sorted(df["cluster"].unique())
        },
        category_orders={
            "Cluster": [nombres_cluster[c] for c in sorted(df["cluster"].unique())]
        },
        custom_data=[
            "id_cliente", "sector", "region", "consumo_kwh",
            "factor_potencia", "antiguedad_anios", "silueta",
        ],
    )
    # El sector no se codifica con simbolo: multiplicaria la leyenda por tres
    # (cuatro clusteres x tres sectores = doce entradas) para una variable que
    # el tooltip ya reporta y que ademas no define la particion.
    fig.update_traces(
        marker=dict(size=9, opacity=0.78, line=dict(width=0.6, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} | %{customdata[2]}<br>"
            "<br>Consumo: %{customdata[3]:,.0f} kWh/mes"
            "<br>Factor de potencia: %{customdata[4]:.3f}"
            "<br>Antiguedad: %{customdata[5]:.1f} anios"
            "<br><br>PC1 = %{x:.2f} | PC2 = %{y:.2f}"
            "<br>Silueta: %{customdata[6]:.3f}"
            "<extra></extra>"
        ),
    )

    # Vectores de carga escalados al rango de las puntuaciones. La flecha se
    # dibuja con una anotacion (que ya trae punta) y el texto va en una segunda
    # anotacion sin flecha, desplazada mas alla de la punta: si se usan las dos
    # cosas en la misma anotacion, Plotly ancla el texto en la cola.
    escala = 0.78 * max(abs(datos["PC1"]).max(), abs(datos["PC2"]).max())
    colocadas = []
    separacion = 0.24 * escala
    orden = cargas.reindex(
        cargas["PC1"].abs().add(cargas["PC2"].abs()).sort_values(ascending=False).index
    )

    for variable in orden.index:
        x = cargas.loc[variable, "PC1"] * escala
        y = cargas.loc[variable, "PC2"] * escala
        fig.add_annotation(
            x=x, y=y, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
            text="", showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=2,
            arrowcolor=AZUL_UNISALLE, opacity=0.9,
        )

        # Mismo escalonado perpendicular que en la version estatica: seis de las
        # nueve variables son casi colineales.
        norma = np.hypot(x, y) or 1.0
        perp = (-y / norma, x / norma)
        base = (x * 1.10, y * 1.10)
        destino = base
        for paso in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
            candidato = (base[0] + perp[0] * separacion * paso,
                         base[1] + perp[1] * separacion * paso)
            if all(np.hypot(candidato[0] - qx, candidato[1] - qy) >= separacion
                   for qx, qy in colocadas):
                destino = candidato
                break
        colocadas.append(destino)

        fig.add_annotation(
            x=destino[0], y=destino[1], xref="x", yref="y",
            text=ETIQUETAS_CORTAS.get(variable.replace("log_", ""), variable),
            showarrow=False, font=dict(size=11, color=AZUL_UNISALLE),
            bgcolor="rgba(255,255,255,0.85)", borderpad=2,
        )

    fig.update_layout(
        **PLANTILLA_PLOTLY,
        title=titulo_plotly(
            "Biplot interactivo",
            "Pase el cursor sobre un punto para ver el cliente; "
            "haga clic en la leyenda para aislar un grupo",
        ),
        xaxis_title=eje_componente(cargas, varianza, "PC1"),
        yaxis_title=eje_componente(cargas, varianza, "PC2"),
        legend=dict(title="", orientation="v", x=1.01, y=1, font=dict(size=11)),
        height=720,
    )
    fig.update_xaxes(zeroline=True, zerolinecolor=BORDE, gridcolor="#F1F3F8")
    fig.update_yaxes(zeroline=True, zerolinecolor=BORDE, gridcolor="#F1F3F8")
    return guardar(fig, ruta)


# -----------------------------------------------------------------------------
# 2. DISPERSION 3D
# -----------------------------------------------------------------------------
def dispersion_3d(df, nombres_cluster, ruta):
    """Las tres variables del modelo en tres ejes, por cluster.

    Al haber reducido el modelo a tres variables, esta figura no es una
    proyeccion: es el espacio completo sobre el que se calcularon el PCA y los
    grupos. Se usan las variables originales -no las componentes- porque en 3D
    el lector quiere reconocer unidades: kWh, adimensional y anios. El consumo
    va en escala logaritmica, igual que en el modelo.
    """
    datos = df.copy()
    datos["Cluster"] = datos["cluster"].map(nombres_cluster)
    datos["log_consumo"] = np.log10(datos["consumo_kwh"])

    fig = px.scatter_3d(
        datos,
        x="log_consumo", y="factor_potencia", z="antiguedad_anios",
        color="Cluster",
        color_discrete_map={
            nombres_cluster[c]: color_cluster(c) for c in sorted(df["cluster"].unique())
        },
        opacity=0.82,
        custom_data=["id_cliente", "sector", "region", "consumo_kwh"],
    )
    fig.update_traces(
        marker=dict(size=5.5, line=dict(width=0.4, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} | %{customdata[2]}<br>"
            "<br>Consumo: %{customdata[3]:,.0f} kWh/mes"
            "<br>Factor de potencia: %{y:.3f}"
            "<br>Antiguedad: %{z:.1f} anios"
            "<extra></extra>"
        )
    )

    fig.update_layout(
        **PLANTILLA_PLOTLY,
        title=titulo_plotly(
            "Los cuatro segmentos en el espacio completo del modelo",
            "Las tres variables del analisis, sin proyectar. Arrastre para rotar",
        ),
        scene=dict(
            xaxis=dict(title="log10 del consumo (kWh/mes)", backgroundcolor="white",
                       gridcolor=BORDE),
            yaxis=dict(title="Factor de potencia", backgroundcolor="white",
                       gridcolor=BORDE),
            zaxis=dict(title="Antiguedad (anios)", backgroundcolor="white",
                       gridcolor=BORDE),
            camera=dict(eye=dict(x=1.65, y=1.5, z=0.85)),
        ),
        legend=dict(title="", x=0.82, y=0.94, font=dict(size=11)),
        height=760,
    )
    return guardar(fig, ruta)


# -----------------------------------------------------------------------------
# 3. COORDENADAS PARALELAS
# -----------------------------------------------------------------------------
def coordenadas_paralelas(df, nombres_cluster, ruta):
    """Las nueve variables del modelo en ejes paralelos, coloreadas por cluster.

    Cada linea es un cliente. Los ejes se muestran en puntuaciones z para que
    compartan escala; arrastrando un intervalo sobre cualquier eje se filtra el
    resto de la figura, que es lo que convierte la grafica en una herramienta de
    exploracion y no en una ilustracion.
    """
    datos = df[VARIABLES_MODELO].copy()
    for col in [c for c in VARIABLES_LOG if c in VARIABLES_MODELO]:
        datos[col] = np.log(datos[col])
    datos = (datos - datos.mean()) / datos.std()
    # Los valores se recortan a +-3 sigma solo para dibujar: un unico cliente con
    # z = 4.7 en interrupciones estiraria ese eje y aplastaria el resto de lineas.

    # Los ticks van en los enteros interiores y no en los extremos: Plotly rotula
    # aparte los limites del rango, y si un tick cae justo ahi el numero sale
    # impreso dos veces sobre el mismo punto del eje.
    dimensiones = [
        dict(
            label=(f"log {ETIQUETAS_CORTAS[c]}" if c in VARIABLES_LOG
                   else ETIQUETAS_CORTAS[c]),
            values=datos[c].clip(-3, 3),
            range=[-3, 3],
            tickvals=[-2, -1, 0, 1, 2],
        )
        for c in VARIABLES_MODELO
    ]

    clusters = sorted(df["cluster"].unique())
    # Escala discreta: un tramo de color por cluster, con los cortes en los
    # puntos medios para que cada valor entero caiga en su franja.
    escala = []
    for i, c in enumerate(clusters):
        escala.append([i / len(clusters), color_cluster(c)])
        escala.append([(i + 1) / len(clusters), color_cluster(c)])

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=df["cluster"],
                colorscale=escala,
                cmin=-0.5,
                cmax=len(clusters) - 0.5,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Cluster", side="top"),
                    tickvals=clusters,
                    ticktext=[nombres_cluster[c].split(":")[0] for c in clusters],
                    thickness=14, len=0.55, y=0.5,
                ),
            ),
            dimensions=dimensiones,
            labelangle=-18,
            labelfont=dict(size=11, color=AZUL_UNISALLE),
            tickfont=dict(size=9),
        )
    )
    fig.update_layout(
        **{k: v for k, v in PLANTILLA_PLOTLY.items() if k != "margin"},
        margin=dict(l=90, r=80, t=110, b=40),
        title=titulo_plotly(
            "Perfil de los clientes en coordenadas paralelas",
            "Variables en puntuaciones z. Arrastre un intervalo sobre cualquier eje "
            "para filtrar; vuelva a hacer clic para liberarlo",
        ),
        height=600,
    )
    return guardar(fig, ruta)


# -----------------------------------------------------------------------------
# 4. SUNBURST DE COMPOSICION
# -----------------------------------------------------------------------------
def sunburst_composicion(df, nombres_cluster, ruta):
    """Jerarquia cluster > sector > region con el consumo total como area."""
    datos = df.copy()
    datos["Cluster"] = datos["cluster"].map(lambda c: nombres_cluster[c].split(":")[0])
    datos["Segmento"] = datos["cluster"].map(nombres_cluster)

    fig = px.sunburst(
        datos,
        path=["Cluster", "sector", "region"],
        values="consumo_kwh",
        color="Cluster",
        color_discrete_map={
            nombres_cluster[c].split(":")[0]: color_cluster(c)
            for c in sorted(df["cluster"].unique())
        },
        custom_data=["Segmento"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Consumo agregado: %{value:,.0f} kWh/mes<br>"
            "Participacion: %{percentRoot:.1%} del total"
            "<extra></extra>"
        ),
        insidetextorientation="radial",
        marker=dict(line=dict(color="white", width=1.5)),
    )
    fig.update_layout(
        **PLANTILLA_PLOTLY,
        title=titulo_plotly(
            "Composicion de cada segmento: cluster &gt; sector &gt; region",
            "El area es el consumo agregado, no el numero de clientes. "
            "Haga clic en un anillo para profundizar",
        ),
        height=700,
    )
    return guardar(fig, ruta)


# -----------------------------------------------------------------------------
def main():
    if not CLUSTERS_PATH.exists():
        raise SystemExit(
            "Falta data/processed/clientes_con_cluster.csv. "
            "Ejecuta antes: python utils/codes/python/pca_clustering.py"
        )

    df = pd.read_csv(CLUSTERS_PATH)
    nombres_cluster = etiquetas_cluster(df)

    # Se recalcula el PCA para recuperar las cargas, que el CSV no guarda.
    X, nombres, _ = preparar_matriz(df)
    _, _, varianza, cargas = ejecutar_pca(X, nombres)

    print(f"Datos: {len(df)} clientes | k = {df['cluster'].nunique()}\n")

    resultados = [
        ("01_biplot_interactivo",
         biplot_interactivo(df, cargas, varianza, nombres_cluster, "01_biplot_interactivo"),
         "Biplot PC1-PC2 con detalle por cliente"),
        ("02_dispersion_3d",
         dispersion_3d(df, nombres_cluster, "02_dispersion_3d"),
         "Dispersion 3D rotable de los segmentos"),
        ("03_coordenadas_paralelas",
         coordenadas_paralelas(df, nombres_cluster, "03_coordenadas_paralelas"),
         "Coordenadas paralelas con filtros por eje"),
        ("04_sunburst_composicion",
         sunburst_composicion(df, nombres_cluster, "04_sunburst_composicion"),
         "Sunburst cluster > sector > region"),
    ]

    for nombre, estado, descripcion in resultados:
        print(f"  OK  {nombre:26s} {descripcion:44s} [{estado}]")

    print(f"\nOK - Fase 4: {len(resultados)} figuras interactivas en "
          f"{FIGURAS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
