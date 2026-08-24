"""Paleta institucional y estilos compartidos por todas las figuras.

Las cinco fases producen figuras con Matplotlib, Seaborn y Plotly, y el
dashboard las repite en el navegador. Centralizar aqui los colores evita que
cada herramienta imponga su paleta por defecto y que el mismo sector aparezca
azul en una figura y naranja en la siguiente.

Los colores son los de la identidad de la Universidad de La Salle (azul
institucional y dorado) mas una serie categorica accesible para los graficos.
"""

import matplotlib.pyplot as plt

# --- Identidad institucional -------------------------------------------------
AZUL_UNISALLE = "#002D57"
DORADO_UNISALLE = "#FFCD00"

# --- Serie categorica --------------------------------------------------------
AZUL = "#4472C4"
NARANJA = "#ED7D31"
VERDE = "#27AE60"
ROJO = "#C0392B"
MORADO = "#8E44AD"
TURQUESA = "#17A2B8"

TEXTO = "#3D4A5C"
TEXTO_SUAVE = "#94A3B8"
BORDE = "#E1E5EE"

# Orden fijo de sectores y regiones: garantiza el mismo color en Python, R y Dash.
COLOR_SECTOR = {
    "Residencial": AZUL,
    "Comercial": NARANJA,
    "Industrial": AZUL_UNISALLE,
}
ORDEN_SECTOR = ["Residencial", "Comercial", "Industrial"]

COLOR_REGION = {
    "Andina": VERDE,
    "Caribe": DORADO_UNISALLE,
    "Pacifica": TURQUESA,
}
ORDEN_REGION = ["Andina", "Caribe", "Pacifica"]

# Paleta de clusteres: se indexa por numero de cluster (0, 1, 2, ...).
COLOR_CLUSTER = [AZUL, NARANJA, VERDE, MORADO, ROJO, TURQUESA, DORADO_UNISALLE]

# Escala divergente para matrices de correlacion y de cargas (rojo - blanco - azul).
ESCALA_DIVERGENTE = "RdBu_r"
ESCALA_SECUENCIAL = "Blues"

# Nombres legibles para ejes y leyendas.
ETIQUETAS = {
    "consumo_kwh": "Consumo (kWh/mes)",
    "costo_miles_cop": "Costo (miles COP)",
    "area_m2": "Area (m2)",
    "potencia_instalada_kw": "Potencia instalada (kW)",
    "num_equipos": "Numero de equipos",
    "horas_operacion": "Horas de operacion (h/mes)",
    "temperatura_c": "Temperatura (C)",
    "factor_potencia": "Factor de potencia",
    "antiguedad_anios": "Antiguedad (anios)",
    "interrupciones_mes": "Interrupciones (por mes)",
}

# Version corta para ejes apretados (heatmaps, coordenadas paralelas).
ETIQUETAS_CORTAS = {
    "consumo_kwh": "Consumo",
    "costo_miles_cop": "Costo",
    "area_m2": "Area",
    "potencia_instalada_kw": "Potencia",
    "num_equipos": "Equipos",
    "horas_operacion": "Horas oper.",
    "temperatura_c": "Temperatura",
    "factor_potencia": "F. potencia",
    "antiguedad_anios": "Antiguedad",
    "interrupciones_mes": "Interrupciones",
}


def aplicar_estilo_matplotlib():
    """Fija los rcParams comunes a todas las figuras estaticas."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlecolor": AZUL_UNISALLE,
            "axes.labelsize": 10,
            "axes.labelcolor": TEXTO,
            "axes.edgecolor": BORDE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": BORDE,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.6,
            "xtick.color": TEXTO,
            "ytick.color": TEXTO,
            "legend.frameon": False,
            "figure.facecolor": "white",
        }
    )


# Plantilla comun de Plotly: fondo blanco, tipografia y colores del dashboard.
# No incluye ``title`` a proposito: cada figura pasa el suyo con titulo_plotly()
# y ambos no pueden convivir en la misma llamada a update_layout().
PLANTILLA_PLOTLY = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family='"Segoe UI", Arial, sans-serif', size=12, color=TEXTO),
    margin=dict(l=70, r=40, t=100, b=60),
)


def titulo_plotly(texto, subtitulo=None):
    """Compone el titulo de una figura de Plotly con subtitulo opcional.

    Plotly no tiene subtitulo nativo, asi que se incrusta como una segunda linea
    del titulo con su propio tamano y color; centralizarlo aqui evita repetir el
    HTML en cada figura y mantiene el mismo aspecto en las cuatro.
    """
    html = f"<b>{texto}</b>"
    if subtitulo:
        html += (
            f"<br><span style='font-size:12px;color:{TEXTO_SUAVE}'>"
            f"{subtitulo}</span>"
        )
    return dict(text=html, font=dict(size=16, color=AZUL_UNISALLE), x=0.01,
                xanchor="left", y=0.97, yanchor="top")
