# -----------------------------------------------------------------------------
# Actividad 6 - Fase 5: replica del analisis multivariante en R
# Maestria en Inteligencia Artificial - Universidad de La Salle
#
# Este script NO vuelve a generar datos: lee el mismo CSV que produjo la Fase 0
# en Python y rehace el analisis con las herramientas propias de R, para
# comprobar que los resultados no dependen del lenguaje ni de la libreria.
#
#   1. Prepara la matriz igual que en Python: las tres variables del conjunto
#      (consumo en logaritmo, factor de potencia y antiguedad), estandarizadas.
#   2. PCA con stats::prcomp y comparacion de autovalores contra los de
#      scikit-learn, leidos de data/processed/pca_varianza.csv.
#   3. K-Means y jerarquico de Ward con las funciones base, seleccion de k por
#      silueta (paquete cluster) e indice Rand ajustado frente a la particion
#      de Python.
#   4. Cinco figuras con ggplot2 y una interactiva con plotly.
#
# Paquetes: ggplot2, plotly, cluster (todos ya instalados en R 4.6).
#
# Ejecucion (desde cualquier directorio; las rutas se resuelven solas):
#   Rscript utils/codes/R/multivariate.R
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(cluster)
  library(plotly)
})

# --- Rutas del proyecto ------------------------------------------------------
#
# R no expone un equivalente de __file__: con rutas relativas manda getwd(), asi
# que una sesion de RStudio abierta sobre otro proyecto escribiria alli las
# figuras. script_path() recupera la ruta real del archivo en los tres modos de
# ejecucion: Rscript (argumento --file=), source() (variable ofile del marco que
# hace la llamada) y el boton Source de RStudio (rstudioapi).
script_path <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
  }
  for (i in seq_len(sys.nframe())) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) {
      return(normalizePath(ofile, mustWork = FALSE))
    }
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    contexto <- rstudioapi::getSourceEditorContext()
    if (!is.null(contexto) && nzchar(contexto$path)) {
      return(normalizePath(contexto$path, mustWork = FALSE))
    }
  }
  NULL
}

this_file <- script_path()
project_root <- if (is.null(this_file)) {
  normalizePath(getwd(), mustWork = FALSE)
} else {
  # utils/codes/R/multivariate.R -> R -> codes -> utils -> raiz
  dirname(dirname(dirname(dirname(this_file))))
}

dataset_path  <- file.path(project_root, "data", "dataset", "consumo_energia.csv")
processed_dir <- file.path(project_root, "data", "processed")
figures_dir   <- file.path(project_root, "public", "assets", "images",
                           "figures", "r", "multivariate")
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figures_dir, recursive = TRUE, showWarnings = FALSE)

set.seed(42)

# --- Paleta institucional (identica a estilo.py) -----------------------------
AZUL_UNISALLE   <- "#002D57"
DORADO_UNISALLE <- "#FFCD00"
AZUL            <- "#4472C4"
NARANJA         <- "#ED7D31"
VERDE           <- "#27AE60"
MORADO          <- "#8E44AD"
ROJO            <- "#C0392B"
TEXTO           <- "#3D4A5C"
TEXTO_SUAVE     <- "#94A3B8"
BORDE           <- "#E1E5EE"

COLOR_CLUSTER <- c(AZUL, NARANJA, VERDE, MORADO, ROJO)
COLOR_SECTOR  <- c(Residencial = AZUL, Comercial = NARANJA, Industrial = AZUL_UNISALLE)

tema_unisalle <- theme_minimal(base_size = 11) +
  theme(
    plot.title      = element_text(face = "bold", size = 13, colour = AZUL_UNISALLE,
                                   hjust = 0, margin = margin(b = 4)),
    plot.subtitle   = element_text(size = 9.5, colour = TEXTO_SUAVE, hjust = 0,
                                   margin = margin(b = 10)),
    plot.caption    = element_text(size = 8, colour = TEXTO_SUAVE, hjust = 0),
    axis.title      = element_text(colour = TEXTO),
    axis.text       = element_text(colour = TEXTO),
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(colour = BORDE, linewidth = 0.35),
    legend.position = "right",
    legend.title    = element_text(size = 9.5, colour = TEXTO),
    plot.title.position = "plot"
  )
theme_set(tema_unisalle)

# Concepto que mide cada variable, para rotular las componentes por su carga
# dominante en vez de escribir el nombre a mano (el orden puede cambiar).
tema_variable <- c(
  consumo_kwh = "escala de consumo",
  factor_potencia = "calidad de la red",
  antiguedad_anios = "calidad de la red"
)

eje_componente <- function(cargas, var_pct, j) {
  dominante <- names(which.max(abs(cargas[, j])))
  # Las filas de las cargas llevan la etiqueta corta, no el nombre de columna.
  variable <- names(etiquetas_cortas)[match(dominante, etiquetas_cortas)]
  sprintf("PC%d - %s (%.1f %%)", j, tema_variable[[variable]], var_pct[j])
}

guardar <- function(grafico, nombre, ancho = 9, alto = 5.6) {
  ggsave(file.path(figures_dir, nombre), grafico,
         width = ancho, height = alto, dpi = 150, bg = "white")
}

# -----------------------------------------------------------------------------
# 1. PREPARACION DE LA MATRIZ (identica a la de Python)
# -----------------------------------------------------------------------------
datos <- read.csv(dataset_path, stringsAsFactors = FALSE)
cat(sprintf("Dataset: %d clientes x %d columnas\n\n", nrow(datos), ncol(datos)))

# Solo el consumo necesita logaritmo; las otras dos ya son de escala acotada.
variables_log <- c("consumo_kwh")

# Las tres variables numericas del conjunto, una por concepto: cuanto consume el
# cliente, con que calidad electrica y desde hace cuanto.
variables_modelo <- c("consumo_kwh", "factor_potencia", "antiguedad_anios")

etiquetas_cortas <- c(
  consumo_kwh = "log Consumo", factor_potencia = "F. potencia",
  antiguedad_anios = "Antiguedad"
)

matriz <- datos[, variables_modelo]
matriz[variables_log] <- log(matriz[variables_log])
# scale() centra y divide por la desviacion muestral (n-1), igual que
# StandardScaler salvo por ese denominador; la diferencia no altera el PCA
# porque afecta a todas las columnas por igual.
X <- scale(as.matrix(matriz))
colnames(X) <- etiquetas_cortas[variables_modelo]

# -----------------------------------------------------------------------------
# 2. PCA Y VERIFICACION CRUZADA CON scikit-learn
# -----------------------------------------------------------------------------
pca <- prcomp(X, center = FALSE, scale. = FALSE)

# El signo de una componente es arbitrario: si se multiplica un autovector por
# -1 sigue siendo autovector del mismo autovalor. prcomp y scikit-learn resuelven
# esa indeterminacion de forma distinta, asi que sin corregirla el biplot de R
# saldria reflejado respecto al de Python y la comparacion visual no valdria.
# Convenio: la variable con mayor carga absoluta en cada componente queda con
# signo positivo, que es el criterio que usa scikit-learn.
for (j in seq_len(ncol(pca$rotation))) {
  dominante <- which.max(abs(pca$rotation[, j]))
  if (pca$rotation[dominante, j] < 0) {
    pca$rotation[, j] <- -pca$rotation[, j]
    pca$x[, j] <- -pca$x[, j]
  }
}

autovalores <- pca$sdev^2
var_pct     <- 100 * autovalores / sum(autovalores)
var_acum    <- cumsum(var_pct)
# Con tres variables el autovalor medio vale 1 por construccion, de modo que el
# criterio de Kaiser ("autovalor > 1") equivale a "por encima del promedio" y
# descartaria la segunda componente pese a recoger un tercio de la informacion.
# Se retiene por varianza acumulada, igual que en Python, y se reporta Kaiser al
# lado para dejar la discrepancia a la vista.
umbral_varianza <- 80
n_retenidas <- max(2, min(which(var_acum >= umbral_varianza)))
n_por_kaiser <- sum(autovalores > 1)

varianza_r <- data.frame(
  componente             = paste0("PC", seq_along(autovalores)),
  autovalor              = round(autovalores, 4),
  varianza_explicada_pct = round(var_pct, 2),
  varianza_acumulada_pct = round(var_acum, 2),
  criterio_kaiser        = ifelse(autovalores > 1, "Retener", "Descartar"),
  criterio_varianza_acum = ifelse(seq_along(autovalores) <= n_retenidas,
                                  "Retener", "Descartar"),
  stringsAsFactors       = FALSE
)
write.csv(varianza_r, file.path(processed_dir, "pca_varianza_r.csv"), row.names = FALSE)

cat("1. VARIANZA EXPLICADA (R)\n")
print(varianza_r, row.names = FALSE)
cat(sprintf("\nComponentes retenidas (varianza acumulada > %d %%): %d -> %.2f %%\n",
            umbral_varianza, n_retenidas, var_acum[n_retenidas]))
cat(sprintf("El criterio de Kaiser habria retenido %d\n\n", n_por_kaiser))

# Comparacion contra los autovalores de scikit-learn.
ruta_py <- file.path(processed_dir, "pca_varianza.csv")
if (file.exists(ruta_py)) {
  varianza_py <- read.csv(ruta_py, stringsAsFactors = FALSE)
  comparacion <- data.frame(
    componente     = varianza_r$componente,
    autovalor_R    = varianza_r$autovalor,
    autovalor_py   = varianza_py$autovalor,
    diferencia_abs = round(abs(varianza_r$autovalor - varianza_py$autovalor), 6),
    var_pct_R      = varianza_r$varianza_explicada_pct,
    var_pct_py     = varianza_py$varianza_explicada_pct
  )
  write.csv(comparacion, file.path(processed_dir, "verificacion_cruzada_pca.csv"),
            row.names = FALSE)
  cat("2. VERIFICACION CRUZADA DE AUTOVALORES (R vs scikit-learn)\n")
  print(comparacion, row.names = FALSE)
  cat(sprintf("\nDiferencia maxima: %.6f\n\n", max(comparacion$diferencia_abs)))
} else {
  cat("2. VERIFICACION CRUZADA omitida: falta pca_varianza.csv (ejecuta la Fase 2)\n\n")
}

# Cargas = correlacion variable-componente.
cargas <- sweep(pca$rotation, 2, pca$sdev, "*")
cargas_df <- as.data.frame(round(cargas[, seq_len(n_retenidas)], 4))
cargas_df$variable <- rownames(cargas)
write.csv(cargas_df, file.path(processed_dir, "pca_cargas_r.csv"), row.names = FALSE)

cat("3. CARGAS DE LAS COMPONENTES RETENIDAS (R)\n")
print(round(cargas[, seq_len(n_retenidas)], 4))
cat("\n")

# --- Figura 1: scree plot ----------------------------------------------------
scree <- data.frame(
  componente = factor(varianza_r$componente, levels = varianza_r$componente),
  varianza   = var_pct,
  acumulada  = var_acum
)

g_scree <- ggplot(scree, aes(x = componente)) +
  geom_col(aes(y = varianza), fill = AZUL, width = 0.62) +
  geom_line(aes(y = acumulada * max(var_pct) / 100, group = 1),
            colour = AZUL_UNISALLE, linewidth = 0.9) +
  geom_point(aes(y = acumulada * max(var_pct) / 100),
             colour = AZUL_UNISALLE, size = 2.1) +
  geom_text(aes(y = varianza, label = ifelse(varianza >= 1.5,
                                             sprintf("%.1f%%", varianza), "")),
            vjust = -0.7, size = 3, colour = TEXTO) +
  geom_vline(xintercept = n_retenidas + 0.5, colour = DORADO_UNISALLE,
             linewidth = 1.1) +
  scale_y_continuous(
    name = "Varianza explicada (%)",
    sec.axis = sec_axis(~ . * 100 / max(var_pct), name = "Varianza acumulada (%)"),
    expand = expansion(mult = c(0, 0.10))
  ) +
  labs(
    title = "Grafico de sedimentacion (scree plot) - R",
    subtitle = sprintf(
      "Criterio de Kaiser: %d componentes con autovalor > 1, que resumen %.1f %% de la informacion",
      n_retenidas, var_acum[n_retenidas]),
    x = "Componente principal"
  )
guardar(g_scree, "01_scree_varianza.png")

# --- Figura 2: biplot --------------------------------------------------------
puntuaciones <- as.data.frame(pca$x[, 1:2])
puntuaciones$sector <- factor(datos$sector,
                              levels = c("Residencial", "Comercial", "Industrial"))

escala_flechas <- 0.80 * max(abs(pca$x[, 1:2]))
flechas <- data.frame(
  variable = rownames(cargas),
  x = cargas[, 1] * escala_flechas,
  y = cargas[, 2] * escala_flechas,
  stringsAsFactors = FALSE
)

# Sin ggrepel disponible, las etiquetas colineales se separan a mano: se
# escalonan en perpendicular a su propia flecha, igual que en la version de
# Python, tomando el primer hueco libre.
separacion <- 0.24 * escala_flechas
orden <- order(abs(flechas$x) + abs(flechas$y), decreasing = TRUE)
flechas$etiqueta_x <- NA_real_
flechas$etiqueta_y <- NA_real_
colocadas <- matrix(numeric(0), ncol = 2)

for (i in orden) {
  norma <- sqrt(flechas$x[i]^2 + flechas$y[i]^2)
  if (norma == 0) norma <- 1
  perp <- c(-flechas$y[i], flechas$x[i]) / norma
  base <- c(flechas$x[i], flechas$y[i]) * 1.12
  destino <- base
  for (paso in c(0, 1, -1, 2, -2, 3, -3, 4, -4)) {
    candidato <- base + perp * separacion * paso
    libre <- TRUE
    if (nrow(colocadas) > 0) {
      distancias <- sqrt((colocadas[, 1] - candidato[1])^2 +
                         (colocadas[, 2] - candidato[2])^2)
      libre <- all(distancias >= separacion)
    }
    if (libre) { destino <- candidato; break }
  }
  colocadas <- rbind(colocadas, destino)
  flechas$etiqueta_x[i] <- destino[1]
  flechas$etiqueta_y[i] <- destino[2]
}

g_biplot <- ggplot(puntuaciones, aes(PC1, PC2)) +
  geom_hline(yintercept = 0, colour = TEXTO_SUAVE, linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = TEXTO_SUAVE, linewidth = 0.3) +
  geom_point(aes(colour = sector), size = 1.9, alpha = 0.72) +
  geom_segment(data = flechas, aes(x = 0, y = 0, xend = x, yend = y),
               arrow = arrow(length = unit(0.18, "cm"), type = "closed"),
               colour = AZUL_UNISALLE, linewidth = 0.55, inherit.aes = FALSE) +
  geom_segment(data = flechas,
               aes(x = x, y = y, xend = etiqueta_x, yend = etiqueta_y),
               colour = TEXTO_SUAVE, linewidth = 0.25, inherit.aes = FALSE) +
  geom_label(data = flechas, aes(x = etiqueta_x, y = etiqueta_y, label = variable),
             colour = AZUL_UNISALLE, size = 2.9, fontface = "bold",
             linewidth = 0, label.padding = unit(0.12, "lines"),
             fill = alpha("white", 0.85), inherit.aes = FALSE) +
  scale_colour_manual(values = COLOR_SECTOR, name = "Sector") +
  labs(
    title = "Biplot: clientes y variables en el plano principal - R",
    subtitle = sprintf("PC1 (%.1f %%) frente a PC2 (%.1f %%); las flechas son las cargas de cada variable",
                       var_pct[1], var_pct[2]),
    x = eje_componente(cargas, var_pct, 1),
    y = eje_componente(cargas, var_pct, 2)
  )
guardar(g_biplot, "02_biplot_pca.png", ancho = 9.5, alto = 6.4)

# -----------------------------------------------------------------------------
# 3. CLUSTERING
# -----------------------------------------------------------------------------
# Mismo espacio que en Python: componentes retenidas con la varianza igualada,
# para que PC1 no domine la distancia euclidea.
Z <- scale(pca$x[, seq_len(n_retenidas)])

seleccion <- data.frame(k = 2:8, inercia = NA_real_, silueta = NA_real_)
for (i in seq_len(nrow(seleccion))) {
  k <- seleccion$k[i]
  km <- kmeans(Z, centers = k, nstart = 25, iter.max = 100)
  sil <- silhouette(km$cluster, dist(Z))
  seleccion$inercia[i] <- round(km$tot.withinss, 2)
  seleccion$silueta[i] <- round(mean(sil[, "sil_width"]), 4)
}
write.csv(seleccion, file.path(processed_dir, "seleccion_k_r.csv"), row.names = FALSE)

k_optimo <- seleccion$k[which.max(seleccion$silueta)]
cat("4. SELECCION DEL NUMERO DE GRUPOS (R)\n")
print(seleccion, row.names = FALSE)
cat(sprintf("\nk elegido por maxima silueta: %d\n\n", k_optimo))

# --- Figura 3: seleccion de k ------------------------------------------------
g_seleccion <- ggplot(seleccion, aes(k)) +
  geom_line(aes(y = silueta), colour = AZUL_UNISALLE, linewidth = 0.9) +
  geom_point(aes(y = silueta), colour = AZUL_UNISALLE, size = 2.4) +
  geom_point(data = seleccion[seleccion$k == k_optimo, ], aes(y = silueta),
             shape = 21, size = 6, colour = ROJO, stroke = 1.1, fill = NA) +
  annotate("text", x = k_optimo + 0.35, y = max(seleccion$silueta) - 0.006,
           label = sprintf("maximo: k = %d\nsilueta = %.3f",
                           k_optimo, max(seleccion$silueta)),
           hjust = 0, size = 3.1, colour = TEXTO) +
  scale_x_continuous(breaks = seleccion$k) +
  labs(
    title = "Seleccion del numero de grupos por el criterio de la silueta - R",
    subtitle = "Calculada con cluster::silhouette sobre las componentes retenidas y estandarizadas",
    x = "Numero de clusteres (k)", y = "Coeficiente de silueta medio"
  )
guardar(g_seleccion, "03_seleccion_k.png", alto = 4.8)

# --- K-Means definitivo y comparacion con Python -----------------------------
km <- kmeans(Z, centers = k_optimo, nstart = 25, iter.max = 100)
sil <- silhouette(km$cluster, dist(Z))
silueta_media <- mean(sil[, "sil_width"])

# La numeracion de los grupos tampoco significa nada: depende del orden en que
# el algoritmo inicializa los centroides, y ademas R numera desde 1 y Python
# desde 0. Si existe la particion de Python, se renumeran los grupos de R para
# que C0 sea el mismo conjunto de clientes en ambos lenguajes; asi las figuras
# de una y otra fase se pueden poner lado a lado. La correspondencia se elige
# por solapamiento maximo, tomando primero el par con mas clientes en comun.
ruta_clusters_py <- file.path(processed_dir, "clientes_con_cluster.csv")
if (file.exists(ruta_clusters_py)) {
  clusters_py <- read.csv(ruta_clusters_py, stringsAsFactors = FALSE)
  solapamiento <- table(km$cluster, clusters_py$cluster)
  equivalencia <- rep(NA_integer_, nrow(solapamiento))

  restante <- solapamiento
  while (any(!is.na(restante) & restante > 0)) {
    celda <- which(restante == max(restante, na.rm = TRUE), arr.ind = TRUE)[1, ]
    equivalencia[celda["row"]] <- as.integer(colnames(solapamiento)[celda["col"]])
    restante[celda["row"], ] <- NA
    restante[, celda["col"]] <- NA
  }
  # Los centroides se reordenan igual, para que los indices sigan casando.
  orden_centroides <- order(equivalencia)
  km$centers <- km$centers[orden_centroides, , drop = FALSE]
  km$cluster <- equivalencia[km$cluster]
  jerarquico_offset <- 0L
} else {
  # Sin referencia, se pasa a base 0 para no arrastrar dos convenios distintos.
  km$cluster <- km$cluster - 1L
  jerarquico_offset <- -1L
}

# Jerarquico de Ward sobre la misma matriz.
jerarquico <- cutree(hclust(dist(Z), method = "ward.D2"), k = k_optimo)

# Indice Rand ajustado. No hay mclust ni fossil instalados, asi que se calcula
# a partir de la tabla de contingencia con la formula de Hubert y Arabie.
rand_ajustado <- function(a, b) {
  tabla <- table(a, b)
  suma_ij <- sum(choose(tabla, 2))
  suma_i  <- sum(choose(rowSums(tabla), 2))
  suma_j  <- sum(choose(colSums(tabla), 2))
  total   <- choose(sum(tabla), 2)
  esperado <- suma_i * suma_j / total
  maximo   <- (suma_i + suma_j) / 2
  (suma_ij - esperado) / (maximo - esperado)
}

cat("5. K-MEANS FRENTE A JERARQUICO DE WARD (R)\n")
print(table(kmeans = km$cluster, ward = jerarquico))
cat(sprintf("Indice Rand ajustado: %.4f\n", rand_ajustado(km$cluster, jerarquico)))
cat(sprintf("Silueta media (k = %d): %.4f\n\n", k_optimo, silueta_media))

# El indice Rand ajustado compara particiones, no etiquetas: vale 1 aunque la
# numeracion difiera, y por eso sigue siendo la medida correcta incluso despues
# de haber renumerado los grupos.
if (exists("clusters_py")) {
  ari_py <- rand_ajustado(km$cluster, clusters_py$cluster)
  concordancia <- data.frame(
    comparacion = c("K-Means R vs K-Means Python",
                    "K-Means R vs Ward R",
                    "Silueta media R",
                    "Silueta media Python"),
    valor = c(round(ari_py, 4),
              round(rand_ajustado(km$cluster, jerarquico), 4),
              round(silueta_media, 4),
              round(mean(clusters_py$silueta), 4)),
    stringsAsFactors = FALSE
  )
  write.csv(concordancia, file.path(processed_dir, "verificacion_cruzada_clusters.csv"),
            row.names = FALSE)
  cat("6. CONCORDANCIA CON LA PARTICION DE PYTHON\n")
  print(concordancia, row.names = FALSE)
  cat("\n")
  print(table(R = km$cluster, Python = clusters_py$cluster))
  cat("\n")
}

# --- Figura 4: clusteres en el plano principal -------------------------------
plano <- data.frame(
  PC1 = pca$x[, 1], PC2 = pca$x[, 2],
  cluster = factor(km$cluster),
  sector = datos$sector,
  region = datos$region,
  id_cliente = datos$id_cliente,
  consumo_kwh = datos$consumo_kwh,
  factor_potencia = datos$factor_potencia,
  antiguedad_anios = datos$antiguedad_anios,
  stringsAsFactors = FALSE
)
tamanos <- table(km$cluster)
niveles_cluster <- as.integer(names(tamanos))
etiquetas_leyenda <- sprintf("C%d (n=%d)", niveles_cluster, as.integer(tamanos))

# Los centroides estan en el espacio estandarizado de las componentes; se
# devuelven a la escala de PC1 y PC2 para dibujarlos sobre la nube.
centroides <- as.data.frame(
  sweep(sweep(km$centers[, 1:2], 2, attr(Z, "scaled:scale")[1:2], "*"),
        2, attr(Z, "scaled:center")[1:2], "+")
)
names(centroides) <- c("PC1", "PC2")

g_clusters <- ggplot(plano, aes(PC1, PC2)) +
  geom_hline(yintercept = 0, colour = TEXTO_SUAVE, linewidth = 0.3) +
  geom_vline(xintercept = 0, colour = TEXTO_SUAVE, linewidth = 0.3) +
  geom_point(aes(colour = cluster), size = 2, alpha = 0.75) +
  geom_point(data = centroides, aes(PC1, PC2), shape = 4, size = 5,
             stroke = 1.6, colour = AZUL_UNISALLE, inherit.aes = FALSE) +
  scale_colour_manual(values = COLOR_CLUSTER[seq_len(k_optimo)],
                      labels = etiquetas_leyenda, name = "Cluster") +
  labs(
    title = sprintf("Particion de K-Means con k = %d sobre el plano principal - R", k_optimo),
    subtitle = sprintf("Silueta media %.3f; las aspas marcan los centroides", silueta_media),
    x = eje_componente(cargas, var_pct, 1),
    y = eje_componente(cargas, var_pct, 2)
  )
guardar(g_clusters, "04_clusters_pca.png", ancho = 9, alto = 6)

# --- Figura 5: perfil de los clusteres ---------------------------------------
# Medias por cluster en puntuaciones z sobre la matriz transformada, la misma
# que alimenta el PCA. Con el consumo en escala original la cola industrial
# infla la desviacion tipica y comprime a los demas grupos.
variables_perfil <- variables_modelo
transformada <- datos[, variables_perfil]
transformada[variables_log] <- log(transformada[variables_log])

perfil <- do.call(rbind, lapply(niveles_cluster, function(c) {
  sub <- transformada[km$cluster == c, ]
  medias <- (colMeans(sub) - colMeans(transformada)) /
    apply(transformada, 2, sd)
  data.frame(cluster = c, variable = names(medias), z = as.numeric(medias),
             stringsAsFactors = FALSE)
}))

nombres_variables <- c(
  consumo_kwh = "log Consumo", factor_potencia = "F. potencia",
  antiguedad_anios = "Antiguedad"
)
perfil$variable <- factor(nombres_variables[perfil$variable],
                          levels = nombres_variables[variables_perfil])
# Los grupos van numerados desde 0 tras alinearlos con Python, asi que no se
# pueden usar como indice de tamanos: hay que buscar la posicion con match().
etiquetas_perfil <- sprintf("C%d\n(n=%d)", niveles_cluster, as.integer(tamanos))
perfil$cluster_etiqueta <- factor(
  etiquetas_perfil[match(perfil$cluster, niveles_cluster)],
  levels = rev(etiquetas_perfil)
)

g_perfil <- ggplot(perfil, aes(variable, cluster_etiqueta, fill = z)) +
  geom_tile(colour = "white", linewidth = 0.8) +
  geom_text(aes(label = sprintf("%+.2f", z)), size = 3, colour = TEXTO) +
  scale_fill_gradient2(low = "#2166AC", mid = "#F7F7F7", high = "#B2182B",
                       midpoint = 0, limits = c(-1.5, 1.5), oob = scales::squish,
                       name = "z") +
  labs(
    title = "Perfil de cada cluster en puntuaciones z - R",
    subtitle = "Cada celda indica cuantas desviaciones tipicas se aparta el grupo de la media global",
    x = NULL, y = NULL
  ) +
  theme(axis.text.x = element_text(angle = 28, hjust = 1),
        panel.grid = element_blank())
guardar(g_perfil, "05_perfil_clusters.png", ancho = 7.6, alto = 4)

# -----------------------------------------------------------------------------
# 4. FIGURA INTERACTIVA CON PLOTLY
# -----------------------------------------------------------------------------
plano$etiqueta <- sprintf(
  "<b>%s</b><br>%s | %s<br>Consumo: %s kWh/mes<br>F. potencia: %.3f<br>Antiguedad: %.1f anios<br>PC1 = %.2f | PC2 = %.2f",
  plano$id_cliente, plano$sector, plano$region,
  format(round(plano$consumo_kwh), big.mark = ","),
  plano$factor_potencia, plano$antiguedad_anios, plano$PC1, plano$PC2
)

g_interactivo <- plot_ly(
  plano, x = ~PC1, y = ~PC2, color = ~cluster,
  colors = COLOR_CLUSTER[seq_len(k_optimo)],
  type = "scatter", mode = "markers",
  marker = list(size = 9, opacity = 0.78,
                line = list(width = 0.6, color = "white")),
  text = ~etiqueta, hoverinfo = "text"
) |>
  layout(
    title = list(
      text = paste0(
        "<b>Segmentacion de clientes en el plano principal (R + plotly)</b><br>",
        "<span style='font-size:12px;color:", TEXTO_SUAVE, "'>",
        "Pase el cursor sobre un punto para identificar al cliente</span>"),
      x = 0.01, xanchor = "left", font = list(size = 16, color = AZUL_UNISALLE)
    ),
    xaxis = list(title = eje_componente(cargas, var_pct, 1),
                 gridcolor = BORDE, zerolinecolor = BORDE),
    yaxis = list(title = eje_componente(cargas, var_pct, 2),
                 gridcolor = BORDE, zerolinecolor = BORDE),
    paper_bgcolor = "white", plot_bgcolor = "white",
    legend = list(title = list(text = "Cluster")),
    margin = list(t = 90)
  )

htmlwidgets::saveWidget(
  g_interactivo,
  file.path(figures_dir, "06_clusters_interactivo.html"),
  selfcontained = FALSE, libdir = "06_clusters_interactivo_files"
)

# -----------------------------------------------------------------------------
resultado <- data.frame(
  metrica = c("n_clientes", "n_variables_modelo", "componentes_retenidas",
              "varianza_acumulada_pct", "k_optimo", "silueta_media"),
  valor = c(nrow(datos), length(variables_modelo), n_retenidas,
            round(var_acum[n_retenidas], 2), k_optimo, round(silueta_media, 4))
)
write.csv(resultado, file.path(processed_dir, "resumen_modelo_r.csv"), row.names = FALSE)

cat("OK - Fase 5 (R): 5 figuras ggplot2 + 1 interactiva en\n")
cat(sprintf("   %s\n", file.path("public", "assets", "images", "figures", "r", "multivariate")))
cat("   Tablas de verificacion cruzada en data/processed/*_r.csv\n")
