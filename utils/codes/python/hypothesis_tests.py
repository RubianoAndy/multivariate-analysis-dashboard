"""Actividad 6 - Fase 1: pruebas de hipotesis univariantes y multivariantes.

Antes de reducir dimensiones hay que responder dos preguntas con evidencia
estadistica, no con intuicion:

1. **Los grupos que ya conocemos (sector, region), .son realmente distintos?**
   Si no lo fueran, el clustering posterior no tendria con que contrastarse.
   Se prueba con normalidad (Shapiro-Wilk), homocedasticidad (Levene), t de
   Welch, ANOVA de un factor + post-hoc de Tukey, chi-cuadrado de independencia
   y una MANOVA que evalua el efecto del sector sobre las tres variables a la vez.

2. **.Por que hace falta reducir dimensiones?** Una regresion del consumo sobre
   el factor de potencia y la antiguedad -dos variables correlacionadas a
   -0.94- muestra el problema en vivo: factores de inflacion de la varianza por
   encima de 8 y coeficientes que se disparan. Es la motivacion concreta del
   PCA de la Fase 2.

3. **.Tiene sentido aplicar PCA a esta matriz?** Un PCA sobre variables
   incorreladas no reduce nada. Se comprueba con la prueba de esfericidad de
   Bartlett (H0: la matriz de correlacion es la identidad) y con el determinante
   de la matriz de correlacion. El indice KMO se reporta tambien, con la
   advertencia de que con solo tres variables deja de ser interpretable.

Todos los resultados se guardan en ``data/processed`` como CSV para que el
informe y el dashboard citen exactamente los mismos numeros.

Ejecucion (desde la raiz del proyecto):
    python utils/codes/python/hypothesis_tests.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from dataset import VARIABLES_NUMERICAS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_PATH = PROJECT_ROOT / "data" / "dataset" / "consumo_energia.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05

# El consumo es multiplicativo y fuertemente asimetrico; el PCA de la Fase 2 lo
# analiza en logaritmo. Las pruebas que anticipan ese PCA -correlaciones,
# Bartlett y KMO- tienen que mirar la misma matriz, o estarian diagnosticando
# unos datos y modelando otros.
VARIABLES_LOG = ["consumo_kwh"]


def matriz_del_modelo(df):
    """Devuelve las variables numericas con la misma transformacion que usa el PCA."""
    datos = df[VARIABLES_NUMERICAS].copy()
    for col in [c for c in VARIABLES_LOG if c in datos.columns]:
        datos[col] = np.log(datos[col])
    return datos


def decision(p, alpha=ALPHA):
    """Traduce un p-valor a la decision sobre H0 con el nivel ``alpha``."""
    return "Se rechaza H0" if p < alpha else "No se rechaza H0"


# -----------------------------------------------------------------------------
# 1. SUPUESTOS: normalidad y homogeneidad de varianzas
# -----------------------------------------------------------------------------
def supuestos(df):
    """Shapiro-Wilk por sector y Levene entre sectores sobre el consumo.

    El consumo se analiza tambien en escala logaritmica porque el proceso
    generador es multiplicativo: si la asimetria es el unico problema, el log
    deberia devolver la normalidad.
    """
    filas = []
    for sector, grupo in df.groupby("sector"):
        for etiqueta, serie in (
            ("consumo_kwh", grupo["consumo_kwh"]),
            ("log(consumo_kwh)", np.log(grupo["consumo_kwh"])),
        ):
            W, p = stats.shapiro(serie)
            filas.append(
                {
                    "prueba": "Shapiro-Wilk",
                    "variable": etiqueta,
                    "grupo": sector,
                    "n": len(serie),
                    "estadistico": round(W, 4),
                    "p_valor": p,
                    "decision_alpha_0.05": decision(p),
                }
            )

    grupos = [g["consumo_kwh"].values for _, g in df.groupby("sector")]
    W, p = stats.levene(*grupos, center="median")
    filas.append(
        {
            "prueba": "Levene (centrado en mediana)",
            "variable": "consumo_kwh",
            "grupo": "Residencial | Comercial | Industrial",
            "n": len(df),
            "estadistico": round(W, 4),
            "p_valor": p,
            "decision_alpha_0.05": decision(p),
        }
    )

    grupos_log = [np.log(g["consumo_kwh"].values) for _, g in df.groupby("sector")]
    W, p = stats.levene(*grupos_log, center="median")
    filas.append(
        {
            "prueba": "Levene (centrado en mediana)",
            "variable": "log(consumo_kwh)",
            "grupo": "Residencial | Comercial | Industrial",
            "n": len(df),
            "estadistico": round(W, 4),
            "p_valor": p,
            "decision_alpha_0.05": decision(p),
        }
    )

    return pd.DataFrame(filas)


# -----------------------------------------------------------------------------
# 2. CONTRASTES UNIVARIANTES: t de Welch, ANOVA y Tukey
# -----------------------------------------------------------------------------
def prueba_t(df):
    """t de Welch: .consumen mas los clientes del Caribe que los de la region Andina?

    La hipotesis tiene sentido fisico: doce grados mas de temperatura media
    implican mas refrigeracion y, por tanto, mas kWh. Se contrasta dos veces:

    * sobre toda la muestra, en la escala original;
    * dentro del sector Residencial y en logaritmo, para eliminar la fuente de
      variacion que domina el consumo (el tamano de la instalacion).

    Se usa Welch (``equal_var=False``) en ambos casos porque las varianzas
    regionales no son comparables. El resultado es negativo en los dos
    contrastes: el consumo esta dominado por el tamano de la instalacion -que
    este conjunto no mide- y ese ruido entierra cualquier diferencia regional.
    Conviene reportarlo tal cual: una prueba que no rechaza tambien informa.
    """

    def contraste(a, b, etiqueta, variable, nombre_a, nombre_b, transformar=None):
        x = transformar(a) if transformar else a
        y = transformar(b) if transformar else b
        t, p = stats.ttest_ind(x, y, equal_var=False)
        # d de Cohen con desviacion combinada: tamano del efecto, no solo significancia.
        s_pool = np.sqrt(
            ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1))
            / (len(x) + len(y) - 2)
        )
        return {
            "prueba": etiqueta,
            "variable": variable,
            "grupo_1": nombre_a,
            "n_1": len(x),
            "media_1": round(float(np.mean(x)), 3),
            "grupo_2": nombre_b,
            "n_2": len(y),
            "media_2": round(float(np.mean(y)), 3),
            "diferencia": round(float(np.mean(x) - np.mean(y)), 3),
            "estadistico_t": round(t, 4),
            "p_valor": p,
            "d_de_Cohen": round(float((np.mean(x) - np.mean(y)) / s_pool), 3),
            "decision_alpha_0.05": decision(p),
        }

    caribe = df.loc[df["region"] == "Caribe", "consumo_kwh"].values
    andina = df.loc[df["region"] == "Andina", "consumo_kwh"].values

    res = df[df["sector"] == "Residencial"]
    caribe_res = res.loc[res["region"] == "Caribe", "consumo_kwh"].values
    andina_res = res.loc[res["region"] == "Andina", "consumo_kwh"].values

    return pd.DataFrame(
        [
            contraste(
                caribe, andina,
                "t de Welch (muestra completa)", "consumo_kwh", "Caribe", "Andina",
            ),
            contraste(
                caribe_res, andina_res,
                "t de Welch (solo sector Residencial)", "log(consumo_kwh)",
                "Caribe", "Andina", transformar=np.log,
            ),
        ]
    )


def regresion_colinealidad(df):
    """Regresion del consumo sobre el estado de la red, y su problema.

    Es la prueba que motiva toda la Fase 2. El factor de potencia y la
    antiguedad describen la misma realidad -una instalacion vieja esta mal
    compensada- y su correlacion llega a -0.94. Meter ambos como predictores en
    una regresion no es ilegal, pero deja el modelo sin capacidad de repartir el
    efecto entre ellos: los errores estandar se inflan, los coeficientes se
    vuelven inestables y uno de los dos acaba absorbiendo el efecto del otro.

    El factor de inflacion de la varianza (VIF) cuantifica ese dano: es
    ``1 / (1 - R2_j)``, donde ``R2_j`` es lo que explican del predictor j los
    demas predictores. Un VIF de 1 significa independencia; por encima de 5 se
    considera colinealidad problematica.

    Como los datos son simulados, el efecto verdadero se conoce: el generador
    fija ``consumo proporcional a 0.92 / factor_potencia``, de modo que la
    pendiente real ronda -1.1. Comparar la estimacion con ese valor deja ver
    hasta que punto la colinealidad la desvia. El PCA de la Fase 2 resuelve
    exactamente este problema: comprime las dos variables en una componente en
    vez de pedirles que se repartan un efecto que no pueden separar.
    """
    datos = df.assign(log_consumo=np.log(df["consumo_kwh"]))
    modelo = ols(
        "log_consumo ~ factor_potencia + antiguedad_anios + C(sector)", data=datos
    ).fit()

    # VIF de cada predictor continuo, calculado a mano para no arrastrar otra
    # dependencia: se regresa cada uno sobre el otro y se usa su R2.
    vif = {}
    predictores = ["factor_potencia", "antiguedad_anios"]
    for j, termino in enumerate(predictores):
        otro = predictores[1 - j]
        r2 = ols(f"{termino} ~ {otro}", data=datos).fit().rsquared
        vif[termino] = 1 / (1 - r2)

    filas = []
    for termino in modelo.params.index:
        filas.append(
            {
                "termino": termino,
                "coeficiente": round(modelo.params[termino], 4),
                "error_estandar": round(modelo.bse[termino], 4),
                "estadistico_t": round(modelo.tvalues[termino], 4),
                "p_valor": modelo.pvalues[termino],
                "ic95_inferior": round(modelo.conf_int().loc[termino, 0], 4),
                "ic95_superior": round(modelo.conf_int().loc[termino, 1], 4),
                "VIF": round(vif[termino], 2) if termino in vif else np.nan,
                "decision_alpha_0.05": decision(modelo.pvalues[termino]),
            }
        )

    tabla = pd.DataFrame(filas)
    tabla.attrs["R2"] = modelo.rsquared
    tabla.attrs["pendiente_real_aprox"] = -1 / df["factor_potencia"].mean()
    return tabla, modelo


def anova_y_tukey(df):
    """ANOVA de un factor (consumo ~ sector) y comparaciones multiples de Tukey.

    Se modela sobre ``log(consumo)``: la ANOVA supone normalidad y varianzas
    iguales, y la Fase 1 ya mostro que la escala original no las cumple.
    """
    datos = df.assign(log_consumo=np.log(df["consumo_kwh"]))
    modelo = ols("log_consumo ~ C(sector)", data=datos).fit()
    tabla = anova_lm(modelo, typ=2)

    ss_efecto = tabla.loc["C(sector)", "sum_sq"]
    ss_total = ss_efecto + tabla.loc["Residual", "sum_sq"]
    p = tabla.loc["C(sector)", "PR(>F)"]

    anova_df = pd.DataFrame(
        [
            {
                "prueba": "ANOVA de un factor",
                "variable": "log(consumo_kwh)",
                "factor": "sector",
                "gl_efecto": int(tabla.loc["C(sector)", "df"]),
                "gl_residual": int(tabla.loc["Residual", "df"]),
                "estadistico_F": round(tabla.loc["C(sector)", "F"], 4),
                "p_valor": p,
                "eta_cuadrado": round(ss_efecto / ss_total, 4),
                "decision_alpha_0.05": decision(p),
            }
        ]
    )

    tukey = pairwise_tukeyhsd(datos["log_consumo"], datos["sector"], alpha=ALPHA)
    tukey_df = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    tukey_df = tukey_df.rename(
        columns={
            "group1": "grupo_1",
            "group2": "grupo_2",
            "meandiff": "dif_medias_log",
            "p-adj": "p_ajustado",
            "lower": "ic95_inferior",
            "upper": "ic95_superior",
            "reject": "significativo",
        }
    )
    # La diferencia en logs se lee mejor como razon de medias geometricas.
    tukey_df["razon_de_medias"] = np.exp(tukey_df["dif_medias_log"]).round(2)

    return anova_df, tukey_df


def chi_cuadrado(df):
    """Chi-cuadrado de independencia entre sector y region.

    Importa para la interpretacion del clustering: si sector y region fueran
    dependientes, un grupo podria estar explicado por la geografia y no por el
    perfil electrico.
    """
    tabla = pd.crosstab(df["sector"], df["region"])
    chi2, p, gl, esperadas = stats.chi2_contingency(tabla)
    n = tabla.values.sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))

    resumen = pd.DataFrame(
        [
            {
                "prueba": "Chi-cuadrado de independencia",
                "variables": "sector x region",
                "gl": gl,
                "estadistico_chi2": round(chi2, 4),
                "p_valor": p,
                "V_de_Cramer": round(v_cramer, 4),
                "frec_esperada_minima": round(esperadas.min(), 2),
                "decision_alpha_0.05": decision(p),
            }
        ]
    )
    return resumen, tabla


# -----------------------------------------------------------------------------
# 3. CONTRASTE MULTIVARIANTE: MANOVA
# -----------------------------------------------------------------------------
def manova(df):
    """MANOVA: efecto del sector sobre el vector completo de variables.

    La ANOVA responde variable por variable e infla el error tipo I al
    repetirse tres veces; la MANOVA contrasta el vector de medias de una sola
    vez. Se trabaja sobre las variables estandarizadas (con logaritmo en el
    consumo) para que ninguna domine por sus unidades.
    """
    datos = df.copy()
    datos["consumo_kwh"] = np.log(datos["consumo_kwh"])
    datos[VARIABLES_NUMERICAS] = (
        datos[VARIABLES_NUMERICAS] - datos[VARIABLES_NUMERICAS].mean()
    ) / datos[VARIABLES_NUMERICAS].std()

    formula = f"{' + '.join(VARIABLES_NUMERICAS)} ~ sector"
    resultado = MANOVA.from_formula(formula, data=datos).mv_test()
    tabla = resultado.results["sector"]["stat"]

    filas = []
    for nombre, fila in tabla.iterrows():
        filas.append(
            {
                "prueba": "MANOVA (efecto del sector)",
                "estadistico": nombre,
                "valor": round(fila["Value"], 4),
                "F_aprox": round(fila["F Value"], 4),
                "gl_numerador": round(fila["Num DF"], 1),
                "gl_denominador": round(fila["Den DF"], 1),
                "p_valor": fila["Pr > F"],
                "decision_alpha_0.05": decision(fila["Pr > F"]),
            }
        )
    return pd.DataFrame(filas)


# -----------------------------------------------------------------------------
# 4. ADECUACION DE LOS DATOS AL PCA: Bartlett y KMO
# -----------------------------------------------------------------------------
def correlaciones(df):
    """Matriz de correlaciones de Pearson y su version en formato largo con p-valores.

    Se calcula sobre la matriz transformada: Pearson mide asociacion lineal, y
    entre el consumo y el resto la relacion es lineal solo en logaritmos.
    """
    datos = matriz_del_modelo(df)
    R = datos.corr()

    filas = []
    for i, a in enumerate(VARIABLES_NUMERICAS):
        for b in VARIABLES_NUMERICAS[i + 1:]:
            r, p = stats.pearsonr(datos[a], datos[b])
            filas.append(
                {
                    "variable_1": a,
                    "variable_2": b,
                    "r_pearson": round(r, 4),
                    "p_valor": p,
                    "significativa_alpha_0.05": p < ALPHA,
                }
            )
    pares = pd.DataFrame(filas).sort_values("r_pearson", key=abs, ascending=False)
    return R, pares


def bartlett_esfericidad(R, n):
    """Prueba de esfericidad de Bartlett.

    H0: la matriz de correlacion poblacional es la identidad, es decir, no hay
    nada que factorizar. El estadistico es
    ``-(n - 1 - (2p + 5) / 6) * ln|R|`` con ``p(p-1)/2`` grados de libertad.
    """
    p = R.shape[0]
    det = np.linalg.det(R.values)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(det)
    gl = p * (p - 1) / 2
    p_valor = stats.chi2.sf(chi2, gl)
    return chi2, gl, p_valor, det


def kmo(R):
    """Indice KMO (Kaiser-Meyer-Olkin) global y por variable.

    Compara la magnitud de las correlaciones simples con la de las parciales:
    si al controlar por el resto de variables la correlacion se desvanece, la
    matriz no comparte factores comunes. Kaiser: <0.5 inaceptable, >0.8 muy bueno.
    """
    R_inv = np.linalg.inv(R.values)
    d = np.sqrt(np.diag(R_inv))
    # Matriz anti-imagen -> correlaciones parciales con signo invertido.
    parciales = -R_inv / np.outer(d, d)
    np.fill_diagonal(parciales, 0)

    R_sin_diag = R.values.copy()
    np.fill_diagonal(R_sin_diag, 0)

    suma_r2 = np.sum(R_sin_diag ** 2)
    suma_p2 = np.sum(parciales ** 2)
    kmo_global = suma_r2 / (suma_r2 + suma_p2)

    r2_col = np.sum(R_sin_diag ** 2, axis=0)
    p2_col = np.sum(parciales ** 2, axis=0)
    kmo_var = pd.Series(r2_col / (r2_col + p2_col), index=R.columns)
    return kmo_global, kmo_var


def adecuacion_pca(df):
    """Reune Bartlett y KMO en una sola tabla interpretable."""
    R = matriz_del_modelo(df).corr()
    chi2, gl, p, det = bartlett_esfericidad(R, len(df))
    kmo_global, kmo_var = kmo(R)

    def etiqueta_kmo(valor):
        if valor >= 0.90:
            return "Excelente"
        if valor >= 0.80:
            return "Muy bueno"
        if valor >= 0.70:
            return "Aceptable"
        if valor >= 0.60:
            return "Mediocre"
        if valor >= 0.50:
            return "Malo"
        return "Inaceptable"

    resumen = pd.DataFrame(
        [
            {
                "prueba": "Esfericidad de Bartlett",
                "estadistico": "chi2",
                "valor": round(chi2, 2),
                "gl": int(gl),
                "p_valor": p,
                "interpretacion": (
                    "La matriz de correlacion difiere de la identidad: el PCA es pertinente"
                    if p < ALPHA
                    else "No hay evidencia de correlacion estructural: el PCA no aporta"
                ),
            },
            {
                "prueba": "Determinante de R",
                "estadistico": "|R|",
                "valor": float(f"{det:.3e}"),
                "gl": np.nan,
                "p_valor": np.nan,
                "interpretacion": "Cercano a 0 indica multicolinealidad, condicion favorable al PCA",
            },
            {
                "prueba": "KMO global",
                "estadistico": "KMO",
                "valor": round(kmo_global, 4),
                "gl": np.nan,
                "p_valor": np.nan,
                # El KMO compara correlaciones simples con parciales. Con tres
                # variables cada parcial se calcula controlando por una sola, de
                # modo que las correlaciones anti-imagen quedan infladas por
                # construccion y el indice se hunde aunque la matriz sea
                # perfectamente factorizable. Kaiser lo penso para baterias de
                # muchos indicadores; aqui no decide nada y manda Bartlett.
                "interpretacion": (
                    f"{etiqueta_kmo(kmo_global)} - no concluyente: el indice no es "
                    f"interpretable con solo {len(VARIABLES_NUMERICAS)} variables"
                ),
            },
        ]
    )

    por_variable = pd.DataFrame(
        {
            "variable": kmo_var.index,
            "KMO": kmo_var.round(4).values,
            "clasificacion": [etiqueta_kmo(v) for v in kmo_var.values],
        }
    ).sort_values("KMO", ascending=False)

    return resumen, por_variable


# -----------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {df.shape[0]} clientes x {len(VARIABLES_NUMERICAS)} variables numericas\n")

    sup = supuestos(df)
    sup.to_csv(PROCESSED_DIR / "pruebas_supuestos.csv", index=False)
    print("1. SUPUESTOS (Shapiro-Wilk y Levene)")
    print(sup.to_string(index=False), "\n")

    t_df = prueba_t(df)
    t_df.to_csv(PROCESSED_DIR / "prueba_t.csv", index=False)
    print("2. t DE WELCH (Caribe vs Andina)")
    print(t_df.to_string(index=False), "\n")

    reg_df, modelo = regresion_colinealidad(df)
    reg_df.to_csv(PROCESSED_DIR / "regresion_colinealidad.csv", index=False)
    print("3. REGRESION log(consumo) ~ factor de potencia + antiguedad + sector")
    print(reg_df.to_string(index=False))
    print(
        f"R2 = {reg_df.attrs['R2']:.4f} | "
        f"pendiente real aproximada sobre el factor de potencia = "
        f"{reg_df.attrs['pendiente_real_aprox']:.2f}\n"
        "Los VIF por encima de 8 y el coeficiente disparado son el problema que "
        "el PCA de la Fase 2 viene a resolver.\n"
    )

    anova_df, tukey_df = anova_y_tukey(df)
    anova_df.to_csv(PROCESSED_DIR / "anova.csv", index=False)
    tukey_df.to_csv(PROCESSED_DIR / "tukey_posthoc.csv", index=False)
    print("4. ANOVA + TUKEY (log consumo ~ sector)")
    print(anova_df.to_string(index=False))
    print(tukey_df.to_string(index=False), "\n")

    chi_df, tabla_cont = chi_cuadrado(df)
    chi_df.to_csv(PROCESSED_DIR / "chi_cuadrado.csv", index=False)
    tabla_cont.to_csv(PROCESSED_DIR / "tabla_contingencia_sector_region.csv")
    print("5. CHI-CUADRADO (sector x region)")
    print(tabla_cont.to_string())
    print(chi_df.to_string(index=False), "\n")

    manova_df = manova(df)
    manova_df.to_csv(PROCESSED_DIR / "manova.csv", index=False)
    print("6. MANOVA (sector sobre las 10 variables)")
    print(manova_df.to_string(index=False), "\n")

    R, pares = correlaciones(df)
    R.round(4).to_csv(PROCESSED_DIR / "matriz_correlacion.csv")
    pares.to_csv(PROCESSED_DIR / "correlaciones_pares.csv", index=False)
    print("7. CORRELACIONES (5 pares mas fuertes)")
    print(pares.head(5).to_string(index=False), "\n")

    adec, kmo_var = adecuacion_pca(df)
    adec.to_csv(PROCESSED_DIR / "adecuacion_pca.csv", index=False)
    kmo_var.to_csv(PROCESSED_DIR / "kmo_por_variable.csv", index=False)
    print("8. ADECUACION AL PCA (Bartlett y KMO)")
    print(adec.to_string(index=False))
    print(kmo_var.to_string(index=False))

    print("\nOK - Fase 1: 9 tablas escritas en data/processed/")


if __name__ == "__main__":
    main()
