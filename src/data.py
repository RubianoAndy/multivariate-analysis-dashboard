"""Carga de datos y motor de analisis del dashboard.

Este modulo hace dos cosas.

**Primera: no reimplementa el analisis.** Anade ``utils/codes/python`` al
``sys.path`` e importa de alli ``preparar_matriz`` y ``ejecutar_pca``. El
dashboard ejecuta por tanto exactamente el mismo codigo que produjo las tablas
y las figuras del informe; si manana cambia el criterio de transformacion o la
lista de variables, cambia en un solo sitio y las dos salidas siguen contando lo
mismo. La alternativa -copiar las funciones aqui- garantiza que tarde o
temprano las dos versiones se separen.

**Segunda: recalcula, no filtra.** Cuando el usuario restringe la vista a un
sector o a una region, el dashboard no se limita a esconder puntos del grafico:
vuelve a estandarizar, a extraer componentes y a agrupar sobre el subconjunto
elegido. Es una diferencia de fondo. El PCA de los clientes industriales no es
el PCA global mirado de cerca: las direcciones de maxima varianza dentro de ese
grupo son otras, y el numero de grupos que tiene sentido tambien. Filtrar y
recalcular responden a preguntas distintas, y la que interesa aqui es la
segunda.

El coste de recalcular con 300 clientes es de milisegundos, asi que no hace
falta memoria intermedia.
"""

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

BASE_DIR = Path(__file__).resolve().parents[1]
PUBLIC_DIR = BASE_DIR / 'public'
DATA_DIR = BASE_DIR / 'data'
CODIGOS_DIR = BASE_DIR / 'utils' / 'codes' / 'python'

# El pipeline de analisis es la fuente de verdad; el dashboard lo consume. Los
# nombres que este modulo no usa directamente se reexportan a proposito, para que
# layout.py y callbacks.py los importen de aqui y no vuelvan a tocar sys.path.
sys.path.insert(0, str(CODIGOS_DIR))
from dataset import VARIABLES_NUMERICAS                      # noqa: E402
from pca_clustering import (                                 # noqa: E402,F401
    VARIABLES_LOG, VARIABLES_MODELO,
    preparar_matriz, ejecutar_pca, describir, eje_componente, SEED,
)
from estilo import ETIQUETAS_CORTAS                          # noqa: E402

# Minimo de clientes para que el analisis sea fiable. Muy por encima de las tres
# variables del modelo: con una muestra pequena la matriz de correlacion es
# inestable y el clustering se vuelve ruido.
MINIMO_CLIENTES = 20


def encode_image(path):
    """Codifica una imagen como URI de datos para incrustarla en el HTML.

    Las imagenes viven en ``public/``, que Dash no sirve de forma automatica
    (solo sirve ``assets/``); incrustarlas evita tener que duplicar la carpeta.
    """
    path = Path(path)
    if not path.exists():
        return ''
    datos = base64.b64encode(path.read_bytes()).decode('ascii')
    ext = path.suffix.lstrip('.').lower()
    if ext == 'jpg':
        ext = 'jpeg'
    return f'data:image/{ext};base64,{datos}'


IMAGENES_DIR = PUBLIC_DIR / 'assets' / 'images'
LOGO_SRC = encode_image(IMAGENES_DIR / 'UnisalleDarkLogoV1.png')
AUTHOR_SRC = encode_image(IMAGENES_DIR / 'author' / 'Andy Rubiano.png')

# --- Datos -------------------------------------------------------------------
df = pd.read_csv(DATA_DIR / 'dataset' / 'consumo_energia.csv')

SECTORES = sorted(df['sector'].unique())
REGIONES = sorted(df['region'].unique())

sector_options = [{'label': s, 'value': s} for s in SECTORES]
region_options = [{'label': r, 'value': r} for r in REGIONES]

# Nombres legibles de las variables del modelo, en el orden del modelo.
etiquetas_modelo = [
    f'log {ETIQUETAS_CORTAS[v]}' if v in VARIABLES_LOG else ETIQUETAS_CORTAS[v]
    for v in VARIABLES_MODELO
]

# Resumen del modelo global, para comparar contra la vista filtrada.
_resumen_path = DATA_DIR / 'processed' / 'resumen_modelo.csv'
RESUMEN_GLOBAL = (
    pd.read_csv(_resumen_path).set_index('metrica')['valor'].to_dict()
    if _resumen_path.exists() else {}
)


def filtrar(sectores, regiones):
    """Subconjunto de clientes segun los filtros; listas vacias = sin filtro."""
    datos = df
    if sectores:
        datos = datos[datos['sector'].isin(sectores)]
    if regiones:
        datos = datos[datos['region'].isin(regiones)]
    return datos.reset_index(drop=True)


def nombrar_clusters(perfil_z):
    """Etiquetas legibles de los grupos a partir de su perfil en puntuaciones z.

    Reutiliza ``describir`` de la Fase 2, de modo que el dashboard y el informe
    nombran igual a un grupo con el mismo perfil. Importa que la regla sea
    relativa ademas de absoluta: las z se calculan respecto al subconjunto
    mostrado, asi que al filtrar por un solo sector -donde los clientes ya son
    homogeneos- todos los grupos caerian en la banda neutra y recibirian la
    misma descripcion.

    El nombre es solo comunicacion; no interviene en el modelo.
    """
    tamano = perfil_z['consumo_kwh']
    calidad = (perfil_z['factor_potencia'] - perfil_z['antiguedad_anios']) / 2

    return {
        c: 'C{}: {}, {}'.format(
            c,
            describir(tamano[c], tamano.values, 0.5, -0.3,
                      ('gran consumidor', 'consumidor medio-alto',
                       'consumidor medio-bajo', 'consumidor pequeno')),
            describir(calidad[c], calidad.values, 0.25, -0.25,
                      ('red confiable', 'red aceptable',
                       'red desgastada', 'red degradada')),
        )
        for c in perfil_z.index
    }


def analizar(sectores, regiones, k):
    """Rehace el analisis multivariante completo sobre el subconjunto filtrado.

    Parametros
    ----------
    sectores, regiones : list[str]
        Valores seleccionados en los filtros; lista vacia significa "todos".
    k : int
        Numero de grupos que pide el usuario en el deslizador.

    Retorna
    -------
    dict o None
        Diccionario con los datos enriquecidos, las cargas, la varianza y las
        metricas de validacion. ``None`` si el filtro deja menos de
        ``MINIMO_CLIENTES`` clientes, caso en el que el analisis no es fiable y
        la interfaz muestra un aviso en lugar de una figura enganosa.
    """
    datos = filtrar(sectores, regiones)
    if len(datos) < MINIMO_CLIENTES:
        return None

    X, nombres, escalador = preparar_matriz(datos)
    pca, scores, varianza, cargas = ejecutar_pca(X, nombres)

    # Mismo criterio que la Fase 2: varianza acumulada, no Kaiser. Con tres
    # variables el autovalor medio vale 1 y Kaiser descartaria una componente
    # que recoge un tercio de la informacion.
    n_ret = max(int((varianza['criterio_varianza_acumulada'] == 'Retener').sum()), 2)
    n_ret = min(n_ret, scores.shape[1])
    Z = scores[:, :n_ret] / scores[:, :n_ret].std(axis=0)

    # k no puede superar el numero de clientes disponibles.
    k = int(min(k, len(datos) - 1))
    km = KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(Z)
    etiquetas = km.labels_

    resultado = datos.copy()
    for i in range(scores.shape[1]):
        resultado[f'PC{i + 1}'] = scores[:, i]
    resultado['cluster'] = etiquetas
    resultado['silueta'] = (
        silhouette_samples(Z, etiquetas) if k > 1 else np.zeros(len(datos))
    )

    # Perfil en z respecto al subconjunto mostrado, sobre la matriz transformada
    # -la misma que usa el PCA-. Con el consumo en escala original la cola
    # industrial infla la desviacion tipica y aplasta al resto de los grupos.
    transformada = datos[VARIABLES_NUMERICAS].copy()
    for col in [c for c in VARIABLES_LOG if c in transformada.columns]:
        transformada[col] = np.log(transformada[col])
    transformada['cluster'] = etiquetas

    media = transformada[VARIABLES_NUMERICAS].mean()
    desv = transformada[VARIABLES_NUMERICAS].std().replace(0, np.nan)
    perfil_z = (
        (transformada.groupby('cluster')[VARIABLES_NUMERICAS].mean() - media) / desv
    ).fillna(0)
    perfil = resultado.groupby('cluster')[VARIABLES_NUMERICAS].mean()
    perfil.insert(0, 'n_clientes', resultado.groupby('cluster').size())

    nombres_cluster = nombrar_clusters(perfil_z)
    resultado['nombre_cluster'] = resultado['cluster'].map(nombres_cluster)

    return {
        'datos': resultado,
        'cargas': cargas,
        'varianza': varianza,
        'perfil': perfil,
        'perfil_z': perfil_z,
        'nombres_cluster': nombres_cluster,
        'n_componentes': n_ret,
        'k': k,
        'silueta': float(silhouette_score(Z, etiquetas)) if k > 1 else float('nan'),
        'variables': VARIABLES_MODELO,
        'etiquetas_variables': etiquetas_modelo,
    }
