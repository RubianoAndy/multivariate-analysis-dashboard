<div align="center">
    <img src="public/assets/images/Logo.png" width="250" alt="Logo Universidad de La Salle">
</div>

# Análisis Multivariante y Creación de Dashboards Interactivos

## 📋 Información General

<div align="center">
    <img src="public/assets/images/author/Andy Rubiano.png" width="200" alt="Foto de Andrés Giovanny Rubiano Muñoz" style="border-radius: 10px;">
</div>

| Aspecto | Detalles |
|--------|----------|
| **Autor** | Andrés Giovanny Rubiano Muñoz "Andy Rubiano" |
| **Correo** | arubiano67@unisalle.edu.co |
| **Asignatura** | Ciencia de Datos — Unidad 2, Actividad 6 |
| **Programa** | Maestría en Inteligencia Artificial |
| **Universidad** | Universidad de La Salle |
| **Herramientas** | Python 3.12 (scikit-learn · statsmodels · Matplotlib · Seaborn · Plotly · Dash) y R 4.6 (prcomp · kmeans · ggplot2 · plotly) |
| **Año** | 2026 |
| **Estado** | Completado |

---

## 🎯 Descripción del Proyecto

Segmentación de **300 clientes** de una distribuidora eléctrica colombiana. El conjunto de datos recoge diez variables de consumo, instalación y calidad del suministro, pero **el modelo usa solo tres**: la Fase 2 demuestra con la matriz de correlaciones y el índice KMO que las otras siete eran información repetida o ajena al cliente. El proyecto recorre la cadena completa del análisis multivariante —comprobar supuestos, contrastar hipótesis, seleccionar variables, reducir dimensiones, agrupar y comunicar— y termina en un **dashboard interactivo** que rehace el análisis en vivo sobre el subconjunto que elija quien lo consulta.

### Lo que el análisis encuentra

| Resultado | Cifra |
|---|---|
| Variables del conjunto → variables del modelo | 10 → **3** (redundancia y KMO) |
| Variables del modelo → componentes retenidas | 3 → **2** |
| Información conservada | **98,2 %** de la varianza |
| Grupos descubiertos | **4** (silueta, Calinski-Harabasz y Davies-Bouldin coinciden) |
| Silueta media | **0,471** |
| Concordancia K-Means ↔ Ward | ARI = **0,839** |
| Concordancia Python ↔ R | ARI = **1,000** (partición idéntica) |

Las tres variables son `consumo_kwh`, `factor_potencia` y `antiguedad_anios`: cuánto consume el cliente, con qué calidad eléctrica y desde hace cuánto. Las dos componentes tienen lectura directa: **PC1 (65,5 %) es la calidad de la red** —factor de potencia frente a antigüedad— y **PC2 (32,7 %) es la escala de consumo**. El cruce de ambas produce cuatro segmentos, y uno de ellos —**C2: gran consumidor con red degradada**, 53 clientes que concentran el mayor consumo (4 694 kWh) con el peor factor de potencia (0,82) y la mayor antigüedad (20,9 años)— es el candidato natural a priorizar en un plan de inversión.

### Objetivos

- Aplicar pruebas de hipótesis univariantes y multivariantes (Shapiro-Wilk, Levene, t de Welch, ANCOVA, ANOVA + Tukey, chi-cuadrado, MANOVA) y verificar la adecuación de los datos al PCA (Bartlett y KMO).
- Seleccionar las variables del modelo con criterios cuantitativos, reducir la dimensionalidad con PCA e identificar segmentos con K-Means y agrupamiento jerárquico de Ward, validando la elección de *k* con cuatro índices.
- Construir visualizaciones avanzadas con **Matplotlib, Seaborn y Plotly** en Python y **ggplot2** en R, comparando lo que aporta cada herramienta.
- Publicar los resultados en un **dashboard interactivo** con identidad institucional.
- Verificar de forma cruzada que Python y R llegan al mismo resultado.

---

## 📚 Estructura del Repositorio

```
.
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias de Python
├── app.py                             # Punto de entrada del dashboard (puerto 8050)
├── assets/
│   └── dashboard.css                  # Estilos de los controles del dashboard
├── data/
│   ├── dataset/
│   │   └── consumo_energia.csv        # 300 clientes × 13 columnas (semilla 42)
│   └── processed/                     # 32 tablas de resultados (Python y R)
├── public/
│   └── assets/
│       └── images/
│           ├── Logo.png · UnisalleDarkLogoV1.png · UnisalleLogo.png
│           ├── author/                # Foto del autor
│           ├── screenshots/           # Capturas del dashboard
│           └── figures/
│               ├── python/
│               │   ├── multivariate/  # 6 figuras de PCA y clustering (Matplotlib)
│               │   ├── advanced/      # 6 figuras exploratorias (Seaborn)
│               │   └── interactive/   # 4 figuras interactivas (Plotly, HTML + PNG)
│               └── r/
│                   └── multivariate/  # 5 figuras ggplot2 + 1 interactiva (plotly)
├── src/                               # Dashboard (Dash)
│   ├── theme.py                       # Paleta institucional y componentes
│   ├── data.py                        # Carga y motor de análisis en vivo
│   ├── layout.py                      # Barra lateral y lienzo
│   ├── callbacks.py                   # Recálculo y siete figuras
│   └── dashboard.py                   # Ensamblaje de la aplicación
└── utils/
    └── codes/
        ├── python/
        │   ├── estilo.py              # Paleta compartida por todas las figuras
        │   ├── dataset.py             # Fase 0 · generación del conjunto de datos
        │   ├── hypothesis_tests.py    # Fase 1 · pruebas de hipótesis y adecuación
        │   ├── pca_clustering.py      # Fase 2 · selección de variables, PCA y clustering
        │   ├── advanced_viz.py        # Fase 3 · visualización avanzada (Seaborn)
        │   └── interactive_viz.py     # Fase 4 · visualización interactiva (Plotly)
        └── R/
            └── multivariate.R         # Fase 5 · réplica y verificación cruzada
```

---

## 🧪 Pipeline del Laboratorio

El flujo es **secuencial**: cada fase consume lo que produjo la anterior. Las decisiones de modelado no se toman por costumbre, sino a partir de los diagnósticos de la fase previa.

### Fase 0 · Generación del conjunto de datos

[`dataset.py`](utils/codes/python/dataset.py) simula 300 clientes con **dos factores latentes**, no diez variables independientes: si lo fueran, la matriz de correlación sería la identidad y el PCA no tendría nada que resumir.

- **Escala** — tamaño físico de la instalación (lognormal por sector, con solapamiento deliberado entre sectores). Gobierna área, potencia, equipos, horas, consumo y costo.
- **Calidad** — estado de la red. No es un continuo, sino una **mezcla de dos poblaciones**: el 62 % del parque pasó por un programa de modernización y el resto conserva la infraestructura original. Esa mezcla es la estructura que el clustering debe descubrir sin conocer la etiqueta.

| Salida | Contenido |
|---|---|
| `data/dataset/consumo_energia.csv` | 300 clientes × 13 columnas (2 categóricas, 10 numéricas, 1 identificador) |

### Fase 1 · Pruebas de hipótesis y adecuación al PCA

[`hypothesis_tests.py`](utils/codes/python/hypothesis_tests.py) responde dos preguntas antes de reducir nada.

**¿Los grupos que ya conocemos son realmente distintos?**

| Prueba | Resultado | Decisión (α = 0,05) |
|---|---|---|
| Shapiro-Wilk sobre `consumo_kwh` | p < 10⁻⁵ en los tres sectores | Se rechaza la normalidad |
| Shapiro-Wilk sobre `log(consumo)` | p = 0,31 / 0,41 / 0,96 | **No** se rechaza → se trabaja en logaritmos |
| ANOVA `log(consumo) ~ sector` | F = 984,4 · η² = 0,869 | Se rechaza H₀ |
| Tukey HSD | Los tres pares difieren (p < 0,001) | Industrial consume 6,3× el comercial |
| Chi-cuadrado `sector × región` | χ² = 4,01 · p = 0,404 | **No** se rechaza → son independientes |
| MANOVA (sector sobre las 10 variables del conjunto) | Λ de Wilks = 0,054 · p ≈ 10⁻¹⁶⁸ | Se rechaza H₀ |

**El hallazgo metodológico central.** La t de Welch que compara el consumo del Caribe frente al de la región Andina **no** resulta significativa (p = 0,82), ni siquiera restringida al sector residencial y en logaritmos (p = 0,24). Pero el efecto climático existe: cuando la **ANCOVA** controla la escala del cliente, aparece con nitidez.

```
log(consumo) ~ log(potencia) + log(horas) + temperatura + factor_potencia + sector
temperatura_c:  β = +0,0105  ·  t = 10,47  ·  p ≈ 5 × 10⁻²²  ·  R² = 0,9965
                → +1,06 % de consumo por cada grado adicional
```

Un contraste bivariante puede ocultar una relación que sí existe: la variación entre sectores —cuatro unidades de logaritmo, unas cincuenta veces en kWh— entierra un efecto del 15 %. Es el argumento del análisis multivariante, y aquí no se enuncia, se demuestra.

**¿Tiene sentido aplicar PCA a esta matriz?**

| Diagnóstico | Valor | Lectura |
|---|---|---|
| Esfericidad de Bartlett | χ²(45) = 4 809,7 · p < 0,001 | La matriz difiere de la identidad: el PCA es pertinente |
| Determinante de R | 8,2 × 10⁻⁸ | Multicolinealidad alta, condición favorable |
| KMO global | **0,838** | Muy bueno |
| KMO de `temperatura_c` | **0,456** | **Inaceptable** (umbral de Kaiser: 0,50) |

Ese último valor no se ignora: **decide el diseño del modelo**. La temperatura describe el clima del municipio, no al cliente, y no comparte factores comunes con el resto. Se excluye del PCA y del clustering; su efecto ya quedó cuantificado y aislado en la ANCOVA, y `region` la conserva como atributo categórico.

### Fase 2 · Selección de variables, componentes principales y clustering

[`pca_clustering.py`](utils/codes/python/pca_clustering.py) empieza por decidir **con qué variables** trabajar. Diez columnas no hacen el análisis más riguroso: lo hacen más difícil de leer, y aquí la mayoría son la misma información repetida.

**Paso 1 · Bloques de variables.** La función `bloques_de_variables` agrupa las nueve candidatas por la distancia `1 − |r|`, de modo que dos variables que miden lo mismo quedan a distancia casi cero. El agrupamiento jerárquico encuentra tres bloques:

| Bloque | Variables | r medio interno | Representante |
|---|---|---|---|
| 1 · Tamaño del cliente | consumo, costo, área, potencia, equipos, horas | 0,88 – 0,96 | `consumo_kwh` |
| 2 · Estado de la red | factor de potencia, antigüedad | 0,97 | `factor_potencia` |
| 3 · Calidad del servicio | interrupciones | — | `interrupciones_mes` |

**Paso 2 · Las tres finales.** Se conserva `consumo_kwh` como representante del primer bloque —seis variables con correlaciones sobre 0,90 son una sola— y las dos que definen el segundo, `factor_potencia` y `antiguedad_anios`. Se descarta `interrupciones_mes`, un conteo de Poisson mucho más ruidoso cuya señal ya está en las otras dos, y `temperatura_c` por su KMO de 0,456.

> Las tres miden conceptos distintos: cuánto consume el cliente, con qué calidad eléctrica y desde hace cuánto. Que dos de ellas estén correlacionadas (r = −0,94) no es un defecto del diseño: que las instalaciones viejas tengan mal factor de potencia es un hallazgo del dominio, y comprimirlo en una sola componente es exactamente el trabajo del PCA. De hecho, una selección que evite toda redundancia deja al PCA sin nada que comprimir: las tres variables sin correlacionar que propone el criterio automático solo alcanzan el 83,2 % en dos componentes, frente al 98,2 % de estas.

**Paso 3 · PCA.** Sobre `log(consumo)`, factor de potencia y antigüedad, estandarizadas:

| Componente | Autovalor | Varianza | Interpretación |
|---|---|---|---|
| PC1 | 1,97 | 65,5 % | **Calidad de la red** — factor de potencia (+0,99) frente a antigüedad (−0,97) |
| PC2 | 0,98 | 32,7 % | **Escala de consumo** — log del consumo (+0,97) |
| PC3 | 0,05 | 1,8 % | Residuo: lo que separa a las dos variables de red |

**Aquí el criterio de Kaiser deja de servir.** Sobre matriz de correlaciones el autovalor medio vale 1 por construcción, así que "autovalor > 1" significa "por encima del promedio". Con tres variables, PC2 se queda en 0,98 y Kaiser la descartaría pese a recoger un tercio de la información y ser la única que mide el consumo. Se retiene por **varianza acumulada del 80 %** —dos componentes, 98,2 %— y la tabla sigue reportando Kaiser al lado para dejar la discrepancia a la vista.

**Paso 4 · Una decisión que cambia el resultado.** El clustering no se hace sobre las puntuaciones crudas, sino sobre las **componentes retenidas estandarizadas**. Las crudas heredan la varianza del autovalor (1,97 frente a 0,98), de modo que la distancia euclídea quedaría dominada por PC1 y K-Means partiría a los clientes solo por el estado de su red, ignorando cuánto consumen.

**Paso 5 · Selección de k.** Los cuatro índices se calculan para k = 2…8 y **tres coinciden en k = 4**:

| k | Inercia | Silueta | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|---|
| 2 | 358,4 | 0,401 | 200,9 | 1,087 |
| 3 | 221,3 | 0,419 | 254,2 | 0,805 |
| **4** | **132,1** | **0,471** ↑ | **349,4** ↑ | **0,701** ↓ |
| 5 | 106,6 | 0,444 | 341,3 | 0,747 |
| 6 | 88,1 | 0,426 | 341,9 | 0,770 |

**Validación.** El agrupamiento jerárquico de Ward, con una lógica distinta, llega casi a la misma partición (ARI = 0,839): la estructura es de los datos, no del método.

### Fase 3 · Visualización avanzada con Seaborn

[`advanced_viz.py`](utils/codes/python/advanced_viz.py) produce seis figuras que responden preguntas del analista, no del modelo.

Dos de ellas se dibujan a propósito sobre las **nueve variables candidatas** y no sobre las tres finales: la matriz de correlación y el **clustermap** son la evidencia de por qué sobraban seis. El dendrograma superior del clustermap agrupa las variables solo, sin que nadie le diga cuáles miden lo mismo, y reproduce los bloques que decidieron la selección. Dibujarlas ya reducidas sería enseñar la conclusión y esconder el argumento.

Las otras cuatro trabajan sobre el modelo: matriz de dispersión por grupo (`PairGrid`, ahora seis paneles legibles en vez de veinticinco), perfil de clústeres en puntuaciones z, violines por grupo y la versión gráfica de la ANCOVA.

### Fase 4 · Visualización interactiva con Plotly

[`interactive_viz.py`](utils/codes/python/interactive_viz.py) exporta cuatro figuras en **HTML** —biplot con detalle por cliente, dispersión 3D rotable, coordenadas paralelas con filtros arrastrables y sunburst de composición— y en PNG para el informe.

Las cuatro comparten una única copia de `plotly.min.js` depositada en la carpeta: se abren con doble clic, sin servidor ni conexión, siempre que la carpeta viaje completa. Incrustar la biblioteca en cada archivo los dejaría en 4,7 MB cada uno; enlazarla a un CDN los volvería inservibles sin red.

### Fase 5 · Réplica en R y verificación cruzada

[`multivariate.R`](utils/codes/R/multivariate.R) lee el mismo CSV y rehace todo con `prcomp`, `kmeans`, `hclust` y `cluster::silhouette`.

| Comprobación | Python | R | Diferencia |
|---|---|---|---|
| Autovalor de PC1 | 1,9725 | 1,9659 | 0,0066 |
| Varianza acumulada en 2 componentes | 98,19 % | 98,19 % | 0,00 |
| k elegido por silueta | 4 | 4 | — |
| Silueta media | 0,4706 | 0,4706 | 0,0000 |
| **Partición de los 300 clientes** | — | — | **ARI = 1,000** |

La diferencia en autovalores es el denominador de la desviación típica (*n* frente a *n*−1) y no altera ningún porcentaje. La partición es **idéntica cliente a cliente**: 96 / 63 / 53 / 88 en ambos lenguajes.

Dos indeterminaciones se resuelven de forma explícita para que las figuras de una y otra fase se puedan comparar lado a lado:

- **El signo de cada componente** es arbitrario (si **v** es autovector, −**v** también lo es) y `prcomp` y scikit-learn lo resuelven distinto. Se fija el convenio de que la variable con mayor carga absoluta quede positiva.
- **La numeración de los grupos** depende del orden de inicialización de los centroides, y R numera desde 1 mientras Python lo hace desde 0. Los grupos de R se renumeran por solapamiento máximo con los de Python.

---

## 📊 Los cuatro segmentos

| Grupo | n | Consumo medio | F. potencia | Antigüedad | Interrup./mes |
|---|---:|---:|---:|---:|---:|
| **C0** · consumidor pequeño, red confiable | 96 | 154,6 kWh | 0,94 | 9,5 años | 0,93 |
| **C1** · consumidor pequeño, red degradada | 63 | 179,1 kWh | 0,83 | 21,1 años | 2,68 |
| **C2** · gran consumidor, red degradada | 53 | 4 693,5 kWh | 0,82 | 20,9 años | 2,62 |
| **C3** · gran consumidor, red confiable | 88 | 3 328,1 kWh | 0,93 | 9,6 años | 0,82 |

Tres lecturas que sostiene la evidencia:

1. **Los grupos cruzan la etiqueta de sector, no la reproducen.** C2 y C3 mezclan clientes comerciales e industriales; C0 y C1 dividen a los residenciales en dos poblaciones que la etiqueta administrativa no distinguía. La segmentación aporta información que el sector no tenía.
2. **El modelo generaliza a las variables que no vio.** Los cuatro grupos se separan también en las siete variables descartadas —área, potencia, equipos, horas, costo e interrupciones—, aunque ninguna participó en el ajuste. Y la temperatura permanece plana (z entre −0,08 y +0,12): la partición responde al perfil eléctrico y no a la geografía, coherente con el KMO que motivó excluirla.
3. **C2 es el segmento accionable.** Son 53 clientes —el 18 % del padrón— que combinan el mayor consumo con el peor estado de red: un factor de potencia de 0,82 implica pérdidas por reactiva, y 2,62 interrupciones mensuales sobre los clientes de mayor facturación es donde el costo de la falla es más alto.

## 🖼️ Galería

### El dashboard interactivo

<div align="center">
    <img src="public/assets/images/screenshots/Dashboard_1.png" width="880" alt="Vista general del dashboard">
</div>

**Vista general** — indicadores, plano de componentes con las cargas superpuestas y varianza explicada. La barra lateral filtra por sector y región, ajusta *k* y permite colorear el plano por grupo descubierto, sector declarado o región, que es la forma directa de comprobar si la segmentación reproduce lo que ya se sabía o aporta algo nuevo. Bajo los filtros, la nota metodológica deja a la vista por qué el modelo usa tres variables y no diez.

<div align="center">
    <img src="public/assets/images/screenshots/Dashboard_scroll.png" width="880" alt="El dashboard desplazado: la barra lateral y el encabezado permanecen fijos">
</div>

**La página no se desplaza: lo hace el lienzo.** El dashboard ocupa exactamente la altura de la ventana y se reparte en dos columnas de altura completa; el scroll vive dentro de la derecha. Así la barra lateral —con sus filtros y la firma— queda siempre entera y a mano, y el encabezado se mantiene pegado arriba: da igual a qué altura del análisis se esté, los controles y el título nunca se pierden de vista. La alternativa, dejar crecer el documento, obliga a subir hasta el principio cada vez que se quiere cambiar un filtro.

<div align="center">
    <img src="public/assets/images/screenshots/Dashboard_autor.png" width="470" alt="Ficha del autor en la barra lateral del dashboard">
</div>

**Autoría en la propia herramienta** — el pie de la barra lateral acompaña siempre al análisis, de modo que el dashboard queda firmado sin depender del documento que lo presenta. El encabezado repite la atribución junto al título.

<div align="center">
    <img src="public/assets/images/screenshots/Dashboard_3.png" width="880" alt="El dashboard tras filtrar por sector industrial">
</div>

**El dashboard recalcula, no filtra.** Al restringir la vista al sector industrial no se ocultan puntos: se vuelve a estandarizar, extraer componentes y agrupar sobre esos 57 clientes. El resultado es otro —PC1 sube del 65,5 % al 72,8 % y la silueta mejora de 0,471 a 0,489, porque dentro de un sector homogéneo los grupos quedan más separados—, y los nombres de los segmentos cambian con ellos: lo que a escala global es un "consumidor pequeño" pasa a ser "medio-alto" dentro del industrial. El PCA de un subconjunto no es el PCA global mirado de cerca.

<div align="center">
    <img src="public/assets/images/screenshots/Dashboard_2.png" width="880" alt="Vista completa del dashboard">
</div>

**Vista completa** — todo el lienzo desenrollado en una sola imagen: bajo el plano principal quedan las cargas, los perfiles en z, las coordenadas paralelas, la composición por sector y región, la matriz de correlaciones que justifica la selección de variables y la tabla de perfiles.

### Componentes principales y clustering (Python · Matplotlib)

| | |
|---|---|
| ![Scree plot](public/assets/images/figures/python/multivariate/01_scree_varianza.png) | ![Biplot](public/assets/images/figures/python/multivariate/02_biplot_pca.png) |
| **Sedimentación** — dos componentes acumulan el 98,2 % | **Biplot** — factor de potencia y antigüedad se oponen sobre PC1; el consumo sube por PC2 |
| ![Selección de k](public/assets/images/figures/python/multivariate/03_seleccion_k.png) | ![Clústeres en el plano](public/assets/images/figures/python/multivariate/05_clusters_pca.png) |
| **Selección de k** — el codo sugiere, la silueta decide | **Partición** — los cuatro grupos ocupan un cuadrante cada uno |

| | |
|---|---|
| ![Silueta](public/assets/images/figures/python/multivariate/04_silueta_clusters.png) | ![Dendrograma](public/assets/images/figures/python/multivariate/06_dendrograma.png) |
| **Silueta por cliente** — dónde la asignación es dudosa | **Ward** — la alternativa que no exige fijar *k* de antemano |

### Exploración avanzada (Python · Seaborn)

| | |
|---|---|
| ![Correlaciones](public/assets/images/figures/python/advanced/01_heatmap_correlacion.png) | ![Matriz de dispersión](public/assets/images/figures/python/advanced/02_matriz_dispersion.png) |
| **Correlaciones** — el bloque de seis que justifica quedarse con una | **Dispersión por grupo** — seis paneles legibles, uno por par |

<div align="center">
    <img src="public/assets/images/figures/python/advanced/03_clustermap.png" width="760" alt="Clustermap">
</div>

**Clustermap** — clientes y variables reordenados por similitud. Los bloques aparecen sin imponerlos: la franja de color de la izquierda es la asignación de K-Means y coincide con la que el dendrograma forma por su cuenta.

<div align="center">
    <img src="public/assets/images/figures/python/advanced/04_perfil_clusters.png" width="900" alt="Perfil de clústeres en z">
</div>

**Perfil en puntuaciones z** — la tabla de interpretación convertida en figura, sobre las diez variables. Los grupos se separan también en las siete que no entraron al modelo, y la columna de temperatura permanece plana: la partición no reproduce la geografía.

| | |
|---|---|
| ![Distribuciones](public/assets/images/figures/python/advanced/05_distribuciones_cluster.png) | ![Regresión](public/assets/images/figures/python/advanced/06_regresion_temperatura.png) |
| **Violines** — la forma completa, no solo los cuartiles | **ANCOVA en versión gráfica** — pendientes reales pero pequeñas frente al ruido |

### Figuras interactivas (Python · Plotly)

Se abren con doble clic, sin servidor ni conexión, mientras la carpeta que los contiene viaje completa (comparten `plotly.min.js`).

| Figura | Interacción | Archivo |
|---|---|---|
| ![Biplot interactivo](public/assets/images/figures/python/interactive/01_biplot_interactivo.png) | Del punto atípico al cliente que lo produce | [`01_biplot_interactivo.html`](public/assets/images/figures/python/interactive/01_biplot_interactivo.html) |
| ![Dispersión 3D](public/assets/images/figures/python/interactive/02_dispersion_3d.png) | Rotar para comprobar si los grupos siguen separados fuera del plano | [`02_dispersion_3d.html`](public/assets/images/figures/python/interactive/02_dispersion_3d.html) |
| ![Coordenadas paralelas](public/assets/images/figures/python/interactive/03_coordenadas_paralelas.png) | Arrastrar un intervalo sobre cualquier eje filtra el resto | [`03_coordenadas_paralelas.html`](public/assets/images/figures/python/interactive/03_coordenadas_paralelas.html) |
| ![Sunburst](public/assets/images/figures/python/interactive/04_sunburst_composicion.png) | Clic en un anillo para profundizar | [`04_sunburst_composicion.html`](public/assets/images/figures/python/interactive/04_sunburst_composicion.html) |

### Réplica en R (ggplot2)

| | |
|---|---|
| ![Scree en R](public/assets/images/figures/r/multivariate/01_scree_varianza.png) | ![Biplot en R](public/assets/images/figures/r/multivariate/02_biplot_pca.png) |
| **Sedimentación** — mismos autovalores que scikit-learn | **Biplot** — misma orientación tras fijar el signo de las componentes |
| ![Selección de k en R](public/assets/images/figures/r/multivariate/03_seleccion_k.png) | ![Clústeres en R](public/assets/images/figures/r/multivariate/04_clusters_pca.png) |
| **Silueta con `cluster::silhouette`** — máximo en k = 4 | **Partición** — idéntica a la de Python, cliente a cliente |

<div align="center">
    <img src="public/assets/images/figures/r/multivariate/05_perfil_clusters.png" width="900" alt="Perfil de clústeres en R">
</div>

**Perfil en R** — mismas cifras que la versión de Seaborn, construidas con `geom_tile`. R aporta además su propia figura interactiva, [`06_clusters_interactivo.html`](public/assets/images/figures/r/multivariate/06_clusters_interactivo.html), generada con `plotly` y `htmlwidgets`.

---

## ⚖️ Comparación de herramientas

Las cinco bibliotecas no compiten: cada fase usa la que resuelve su problema con menos fricción.

| Herramienta | Dónde se usa | Por qué esa y no otra |
|---|---|---|
| **Matplotlib** | Fase 2 — figuras del modelo | Control absoluto del lienzo. El biplot exige dibujar flechas desde el origen, escalarlas a la nube de puntos y colocar cada etiqueta fuera de su punta sin que se pisen; nada de eso existe como primitiva en las bibliotecas de alto nivel. |
| **Seaborn** | Fase 3 — exploración | Tres capacidades que en Matplotlib costarían decenas de líneas: mapa de calor anotado con paleta divergente centrada, `PairGrid` por grupos y `clustermap`, que reordena filas y columnas por similitud y dibuja los dendrogramas en los márgenes. |
| **Plotly** | Fase 4 — comunicación | Cambia el contrato: el lector hace zoom, aísla series, rota el espacio y lee los datos de un cliente al pasar el cursor. La interpretación deja de estar cerrada de antemano. |
| **Dash** | Dashboard | Convierte las figuras de Plotly en una aplicación con estado. El valor no es mostrar gráficas, sino **rehacer el análisis** con cada interacción. |
| **ggplot2** | Fase 5 — verificación | La gramática de gráficos hace que replicar una figura sea describirla, no reconstruirla. Al ser una implementación independiente, es una verificación real y no una repetición del mismo código. |

Dos apuntes prácticos encontrados al construirlo:

- **Seaborn no siempre gana.** `clustermap` monta sus ejes sobre un *gridspec* propio que ignora `tight_layout`, y la barra de color es una celda de esa rejilla: hay que bajar el techo a mano y reposicionarla después. Matplotlib puro habría sido más predecible, pero no habría dado los dendrogramas marginales.
- **La ergonomía de Plotly tiene aristas.** Una anotación con flecha ancla el texto en la cola, no en la punta, así que un biplot necesita dos anotaciones por vector. Y si un `tickval` cae justo en el extremo del rango, el número se imprime dos veces sobre el mismo punto del eje.

---

## ⚙️ Requisitos

### Python

> ⚠️ **Versión:** Python 3.10 o superior (probado en **3.12.10**), con entorno virtual dedicado.

| Dependencia | Versión probada | Uso |
|---|---|---|
| `numpy` | 2.5.2 | Cálculo numérico |
| `pandas` | 3.0.5 | Manejo de datos y tablas de resultados |
| `scipy` | 1.18.1 | Shapiro-Wilk, Levene, t de Welch, chi-cuadrado, dendrograma |
| `statsmodels` | 0.14.6 | ANOVA, ANCOVA, Tukey y MANOVA |
| `scikit-learn` | 1.9.0 | PCA, K-Means, jerárquico y métricas de validación |
| `matplotlib` | 3.11.1 | Figuras de la Fase 2 |
| `seaborn` | 0.13.2 | Figuras de la Fase 3 |
| `plotly` | 6.9.0 | Figuras interactivas y del dashboard |
| `dash` | 4.4.1 | Dashboard |
| `kaleido` | 1.3.0 | Exportación de las figuras de Plotly a PNG |

### R

- **R 4.x** (probado en **4.6.1**) con `ggplot2`, `cluster` y `plotly`. Sin dependencias adicionales: `prcomp`, `kmeans` y `hclust` son de la instalación base.
- Editor: RStudio Desktop o VS Code con la extensión **R** (REditorSupport) + `languageserver`.

---

## 🛠️ Ejecución

Todos los comandos se lanzan **desde la raíz del proyecto**.

```bash
# 1. Entorno de Python
python -m venv .venv
source .venv/Scripts/activate        # Git Bash (en PowerShell: .venv\Scripts\activate)
pip install -r requirements.txt

# 2. Pipeline de análisis, en orden
python utils/codes/python/dataset.py            # Fase 0 · dataset
python utils/codes/python/hypothesis_tests.py   # Fase 1 · pruebas de hipótesis
python utils/codes/python/pca_clustering.py     # Fase 2 · PCA y clustering
python utils/codes/python/advanced_viz.py       # Fase 3 · figuras de Seaborn
python utils/codes/python/interactive_viz.py    # Fase 4 · figuras de Plotly

# 3. Réplica en R y verificación cruzada
Rscript utils/codes/R/multivariate.R            # Fase 5

# 4. Dashboard interactivo
python app.py                                   # http://localhost:8050/
```

> Las fases 3, 4 y 5 leen `data/processed/clientes_con_cluster.csv`, así que la Fase 2 debe haberse ejecutado antes. Los scripts avisan y se detienen si falta.

**Notas de reproducibilidad y de entorno**

- **Semilla fija** (`default_rng(42)` en Python, `set.seed(42)` en R): cualquier ejecución reproduce el mismo CSV, las mismas figuras y las mismas cifras de este README.
- **Rutas**: Python las resuelve desde la ubicación del script (`Path(__file__)`) y R con un `script_path()` que funciona en `Rscript`, `source()` y el botón *Source* de RStudio. Ninguno depende del directorio de trabajo.
- **La carpeta de estilos del dashboard** se declara de forma explícita (`assets_folder`), porque Dash la busca junto al módulo donde se crea la aplicación —`src/`— y aquí vive en la raíz.
- **Si `kaleido` no encuentra un navegador**, los scripts avisan y continúan: el HTML interactivo es el entregable real y el PNG solo alimenta el informe.

---

## 🔑 Palabras Clave

`Análisis Multivariante` · `PCA` · `K-Means` · `Clustering Jerárquico` · `Pruebas de Hipótesis` · `MANOVA` · `ANCOVA` · `KMO y Bartlett` · `Dashboard Interactivo` · `Dash` · `Plotly` · `Seaborn` · `ggplot2` · `scikit-learn` · `Ciencia de Datos`

---

## 📧 Contacto

**Andrés Giovanny Rubiano Muñoz**
Maestría en Inteligencia Artificial · Universidad de La Salle
arubiano67@unisalle.edu.co

---

## 📄 Derechos Reservados

© 2026 Andrés Giovanny Rubiano Muñoz (Andy Rubiano). Todos los derechos reservados.

Este laboratorio y su contenido —código, datos y documentación— son propiedad intelectual conjunta de:

- **Andrés Giovanny Rubiano Muñoz** (Andy Rubiano) — Autor
- **Universidad de La Salle** — Institución académica

El uso, reproducción o distribución requiere autorización previa escrita de los titulares de derechos.

---

<div align="center">
  Universidad de La Salle | Bogotá D. C., Colombia
</div>
