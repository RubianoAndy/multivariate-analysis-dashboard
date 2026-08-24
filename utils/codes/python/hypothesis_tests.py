"""Actividad 6 - Fase 1: pruebas de hipotesis univariantes y multivariantes.

Antes de reducir dimensiones hay que responder dos preguntas con evidencia
estadistica, no con intuicion:

1. **Los grupos que ya conocemos (sector, region), .son realmente distintos?**
   Si no lo fueran, el clustering posterior no tendria con que contrastarse.
   Se prueba con normalidad (Shapiro-Wilk), homocedasticidad (Levene), t de
   Welch, ANOVA de un factor + post-hoc de Tukey, chi-cuadrado de independencia
   y una MANOVA que evalua el efecto del sector sobre las 10 variables a la vez.

2. **.Tiene sentido aplicar PCA a esta matriz?** Un PCA sobre variables
   incorreladas no reduce nada. Se comprueba con la prueba de esfericidad de
   Bartlett (H0: la matriz de correlacion es la identidad) y con el indice
   KMO de adecuacion muestral, calculado a partir de la matriz de
   correlaciones parciales (anti-imagen).

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
    regionales no son comparables. El resultado -no significativo en los dos
    contrastes- es el que motiva la ANCOVA de la funcion siguiente.
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


def ancova_temperatura(df):
    """ANCOVA: efecto de la temperatura una vez controlada la escala del cliente.

    La t de Welch no detecta el efecto climatico porque el consumo esta
    dominado por el tamano de la instalacion: un residencial del Caribe y un
    industrial andino difieren en dos ordenes de magnitud, y ese ruido entierra
    un efecto del orden del 15 %. La escala actua como variable de confusion.

    El modelo ``log(consumo) ~ log(potencia) + log(horas) + temperatura + sector``
    aisla el efecto parcial de la temperatura manteniendo constante el resto.
    Es el argumento central de la actividad: en datos multivariantes, un
    contraste bivariante puede ocultar una relacion que si existe.
    """
    datos = df.assign(
        log_consumo=np.log(df["consumo_kwh"]),
        log_potencia=np.log(df["potencia_instalada_kw"]),
        log_horas=np.log(df["horas_operacion"]),
    )
    modelo = ols(
        "log_consumo ~ log_potencia + log_horas + temperatura_c + factor_potencia + C(sector)",
        data=datos,
    ).fit()

    filas = []
    for termino in modelo.params.index:
        filas.append(
            {
                "termino": termino,
                "coeficiente": round(modelo.params[termino], 5),
                "error_estandar": round(modelo.bse[termino], 5),
                "estadistico_t": round(modelo.tvalues[termino], 4),
                "p_valor": modelo.pvalues[termino],
                "ic95_inferior": round(modelo.conf_int().loc[termino, 0], 5),
                "ic95_superior": round(modelo.conf_int().loc[termino, 1], 5),
                "decision_alpha_0.05": decision(modelo.pvalues[termino]),
            }
        )

    tabla = pd.DataFrame(filas)
    # Efecto practico: cuanto sube el consumo por cada grado adicional.
    beta_temp = modelo.params["temperatura_c"]
    tabla.attrs["R2"] = modelo.rsquared
    tabla.attrs["efecto_pct_por_grado"] = (np.exp(beta_temp) - 1) * 100
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
    repetirse diez veces; la MANOVA contrasta el vector de medias de una sola
    vez. Se trabaja sobre las variables estandarizadas (log en las de escala)
    para que ninguna domine por sus unidades.
    """
    escala = ["consumo_kwh", "costo_miles_cop", "area_m2", "potencia_instalada_kw",
              "num_equipos", "horas_operacion"]
    datos = df.copy()
    for col in escala:
        datos[col] = np.log(datos[col])
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
    """Matriz de correlaciones de Pearson y su version en formato largo con p-valores."""
    R = df[VARIABLES_NUMERICAS].corr()

    filas = []
    for i, a in enumerate(VARIABLES_NUMERICAS):
        for b in VARIABLES_NUMERICAS[i + 1:]:
            r, p = stats.pearsonr(df[a], df[b])
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
    R = df[VARIABLES_NUMERICAS].corr()
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
                "interpretacion": etiqueta_kmo(kmo_global),
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

    ancova_df, modelo = ancova_temperatura(df)
    ancova_df.to_csv(PROCESSED_DIR / "ancova_temperatura.csv", index=False)
    print("3. ANCOVA (temperatura controlando escala y sector)")
    print(ancova_df.to_string(index=False))
    print(
        f"R2 = {ancova_df.attrs['R2']:.4f} | "
        f"efecto de la temperatura = "
        f"{ancova_df.attrs['efecto_pct_por_grado']:+.2f} % de consumo por grado C\n"
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

    print("\nOK - Fase 1: 10 tablas escritas en data/processed/")


if __name__ == "__main__":
    main()
