"""Actividad 6 - Fase 0: generacion del conjunto de datos multivariante.

Extiende el caso de estudio que vienen usando las actividades anteriores
(consumo de energia de clientes de una distribuidora colombiana) a un conjunto
de **10 variables numericas** y 2 categoricas, que es lo que hace posible
aplicar PCA y clustering: con dos o tres columnas no hay estructura latente que
reducir ni perfiles que descubrir.

El generador no sortea las 10 variables de forma independiente, porque entonces
la matriz de correlacion seria practicamente la identidad y el PCA no tendria
nada que resumir. En su lugar se simulan dos factores latentes que si existen en
el dominio:

* ``escala``     - tamano fisico de la instalacion del cliente. Gobierna area,
                   potencia instalada, numero de equipos, horas de operacion,
                   consumo y costo. Es el factor que el PCA deberia recuperar
                   como primera componente.
* ``calidad``    - condiciones de operacion del suministro. Gobierna el factor
                   de potencia, la antiguedad de la instalacion y las
                   interrupciones. Deberia aparecer como segunda componente,
                   ortogonal a la primera.

A diferencia de la escala, la calidad **no** es un continuo: la distribuidora
ejecuto un programa de modernizacion que renovo parte del parque y dejo el
resto sin intervenir, de modo que ``calidad`` se simula como una mezcla de dos
poblaciones (red renovada / red heredada). Esa mezcla es la estructura real que
el clustering de la Fase 2 debe descubrir sin conocer la etiqueta; el sector,
en cambio, si es una variable observada y sirve de contraste externo.

Semilla fija (``default_rng(42)``): cualquier ejecucion reproduce exactamente el
mismo CSV, de modo que Python, R y el dashboard analizan los mismos numeros.

Rutas: el script se ubica en python -> codes -> utils -> raiz del proyecto y
escribe el CSV en ``data/dataset``.

Ejecucion (desde la raiz del proyecto):
    python utils/codes/python/dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "dataset"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N = 300

SECTORES = ["Residencial", "Comercial", "Industrial"]
PROB_SECTOR = [0.50, 0.30, 0.20]

REGIONES = ["Andina", "Caribe", "Pacifica"]
PROB_REGION = [0.50, 0.30, 0.20]

# Media y desviacion (en escala logaritmica) del factor latente de tamano.
# Los tres sectores se solapan a proposito: el clustering no debe reducirse a
# "un cluster por sector", tiene que descubrir grupos que cruzan la etiqueta.
ESCALA_LOG = {
    "Residencial": (0.00, 0.35),
    "Comercial": (1.15, 0.40),
    "Industrial": (2.30, 0.45),
}

TEMP_BASE = {"Andina": 17.0, "Caribe": 29.0, "Pacifica": 26.0}

# Cobertura del programa de modernizacion de red (factor latente de calidad).
PROP_RED_RENOVADA = 0.62


def generar_dataset(n=N, seed=SEED):
    """Simula ``n`` clientes con estructura latente de dos factores.

    Parametros
    ----------
    n : int
        Numero de clientes a generar.
    seed : int
        Semilla del generador; fija la reproducibilidad del CSV.

    Retorna
    -------
    pandas.DataFrame
        Tabla con 2 columnas categoricas, 10 numericas y el identificador.
    """
    rng = np.random.default_rng(seed)

    sector = rng.choice(SECTORES, size=n, p=PROB_SECTOR)
    region = rng.choice(REGIONES, size=n, p=PROB_REGION)

    # --- Factor latente 1: escala de la instalacion --------------------------
    mu = np.array([ESCALA_LOG[s][0] for s in sector])
    sigma = np.array([ESCALA_LOG[s][1] for s in sector])
    escala = np.exp(rng.normal(mu, sigma))

    # La potencia instalada y las horas de operacion son magnitudes latentes: el
    # consumo sale de multiplicarlas, pero el sistema comercial de la
    # distribuidora no las registra cliente a cliente, asi que no se exportan.
    potencia_kw = 4.2 * escala * rng.lognormal(0, 0.20, n)

    horas_base = {"Residencial": 110, "Comercial": 260, "Industrial": 480}
    horas_operacion = np.array(
        [rng.normal(horas_base[s], 45) for s in sector]
    ).clip(40, 720)

    # --- Covariable climatica ------------------------------------------------
    temperatura_c = np.array([rng.normal(TEMP_BASE[r], 2.0) for r in region])

    # --- Factor latente 2: calidad de la operacion ---------------------------
    # Mezcla de dos poblaciones, no un continuo: el 62 % del parque paso por el
    # programa de modernizacion (red renovada) y el 38 % restante conserva la
    # infraestructura original. Valores altos de ``calidad`` = instalacion
    # moderna, bien compensada y con pocas fallas.
    renovada = rng.random(n) < PROP_RED_RENOVADA
    calidad = np.where(
        renovada,
        rng.normal(0.85, 0.45, n),
        rng.normal(-1.35, 0.50, n),
    )
    antiguedad_anios = np.clip(14 - 5.5 * calidad + rng.normal(0, 1.6, n), 0.5, 40)
    factor_potencia = np.clip(
        0.905 + 0.050 * calidad - 0.012 * np.log(escala + 1) + rng.normal(0, 0.012, n),
        0.60,
        0.99,
    )
    # --- Variable de resultado ---------------------------------------------
    # El consumo es potencia x horas, corregido por el clima (refrigeracion) y
    # penalizado cuando el factor de potencia es bajo.
    consumo_kwh = (
        potencia_kw
        * horas_operacion
        * 0.32
        * (1 + 0.012 * (temperatura_c - 20))
        * (0.92 / factor_potencia)
        * rng.lognormal(0, 0.10, n)
    ).clip(30, None)

    # El conjunto observado se queda en TRES variables numericas, una por
    # concepto: cuanto consume el cliente, con que calidad electrica y desde
    # hace cuanto. Todo lo demas -area, potencia, equipos, horas, temperatura-
    # participa en la simulacion pero no se exporta: son las magnitudes latentes
    # que generan el consumo, no medidas que la distribuidora tenga en su
    # sistema comercial. Un conjunto de diez columnas donde seis miden lo mismo
    # no hace el analisis mas riguroso, solo mas dificil de leer.
    df = pd.DataFrame(
        {
            "id_cliente": [f"CL-{i:04d}" for i in range(1, n + 1)],
            "sector": sector,
            "region": region,
            "consumo_kwh": consumo_kwh.round(1),
            "factor_potencia": factor_potencia.round(3),
            "antiguedad_anios": antiguedad_anios.round(1),
        }
    )
    return df


# Las tres variables numericas del conjunto. Se exporta para que el resto de
# fases (y el dashboard) no repitan la lista.
VARIABLES_NUMERICAS = [
    "consumo_kwh",
    "factor_potencia",
    "antiguedad_anios",
]


def main():
    df = generar_dataset()
    salida = DATA_DIR / "consumo_energia.csv"
    df.to_csv(salida, index=False)

    print(f"Dataset generado: {len(df)} clientes x {df.shape[1]} columnas "
          f"({len(VARIABLES_NUMERICAS)} numericas, 2 categoricas, 1 identificador)")
    print("\nDistribucion por sector:")
    print(df["sector"].value_counts().to_string())
    print("\nMedias por sector:")
    print(df.groupby("sector")[VARIABLES_NUMERICAS].mean().round(3).to_string())
    print(f"\nOK - Fase 0: {salida.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
