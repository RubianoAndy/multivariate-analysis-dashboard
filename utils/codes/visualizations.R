#' Replica en R las figuras de la actividad y verifica los estadisticos.
#'
#' Lee el CSV generado por visualizations.py y escribe las imagenes en
#' public/assets/images/figures/r/.

#' Las rutas se resuelven desde la ubicacion de este archivo, no desde el
#' directorio de trabajo, de modo que las salidas caen siempre dentro de este
#' proyecto aunque la sesion de RStudio apunte a otro.
#'
#' R no expone un equivalente de __file__: con rutas relativas manda getwd(),
#' asi que una sesion abierta sobre otro proyecto escribe alli las figuras.
#' script_path() recupera la ruta real del archivo en los tres modos de
#' ejecucion: Rscript (argumento --file=), source() (variable ofile del marco
#' que hace la llamada) y el boton Source/Run de RStudio (rstudioapi).
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
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
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
  # utils/codes/visualizations.R -> utils/codes -> utils -> raiz del proyecto
  dirname(dirname(dirname(this_file)))
}

#' Rutas del proyecto.
data_path <- file.path(project_root, "data", "dataset", "consumo_energia.csv")
figures_dir <- file.path(project_root, "public", "assets", "images", "figures",
                         "r")
good_dir <- file.path(figures_dir, "good_design")
bad_dir <- file.path(figures_dir, "bad_design")

#' Verificar el dataset antes de crear nada: si la raiz deducida fuera la
#' equivocada, el script se detiene en vez de sembrar carpetas y figuras en
#' otro proyecto.
if (!file.exists(data_path)) {
  stop(sprintf(paste0("No se encontro el dataset en '%s'. Ejecuta antes: ",
                      "python utils/codes/visualizations.py"),
               data_path))
}

#' Crea las carpetas de figuras si no existen.
for (d in c(good_dir, bad_dir)) {
  if (!dir.exists(d)) {
    dir.create(d, recursive = TRUE)
  }
}

cat(sprintf("Raiz del proyecto: %s\n", project_root))

#' Carga de datos y vista rapida de su estructura.
df <- read.csv(data_path)

head(df)
str(df)

#' GRAFICAS BIEN DISEÑADAS (good_design).
#'
#' Principios aplicados: título informativo, ejes con unidades, cuadricula sutil
#' detras de los datos, colores sobrios y etiquetas de datos.

#' Boxplot: consumo por sector.
#'
#' Se dibuja en dos pasadas: la primera establece ejes y escala sin trazar las
#' cajas, y tras insertar la cuadricula horizontal la segunda dibuja las cajas
#' encima.
png(file.path(good_dir, "boxplot_consumption_by_sector.png"),
    width = 1950, height = 1080, res = 300)
par(mar = c(4, 4, 3, 1))
box_colors <- c("#74a9cf", "#2b8cbe", "#a6bddb")
boxplot(consumo_kwh ~ sector, data = df, border = NA,
        main = "Consumo mensual por sector (R)",
        xlab = "Sector", ylab = "Consumo (kWh/mes)")
grid(nx = NA, ny = NULL)
boxplot(consumo_kwh ~ sector, data = df,
        col = box_colors, add = TRUE, axes = FALSE)
dev.off()

#' Dispersion: consumo vs costo.
#'
#' La primera pasada traza un lienzo vacio con ejes para dejar la cuadricula
#' completa detras; la segunda agrega puntos, tendencia y leyenda.
png(file.path(good_dir, "scatter_consumption_vs_cost.png"),
    width = 1950, height = 1140, res = 300)
par(mar = c(4, 4, 3, 1))
sector_colors <- c(Residencial = "#1b9e77",
                   Comercial = "#d95f02",
                   Industrial = "#7570b3")
plot(df$consumo_kwh, df$costo_miles_cop, type = "n",
     main = "Consumo vs. costo mensual (R)",
     xlab = "Consumo (kWh/mes)", ylab = "Costo (miles de COP)")
grid()
points(df$consumo_kwh, df$costo_miles_cop,
       col = sector_colors[df$sector], pch = 19, cex = 0.6)
abline(lm(costo_miles_cop ~ consumo_kwh, data = df), lty = 2)
legend("topleft", legend = names(sector_colors),
       col = sector_colors, pch = 19, cex = 0.8)
dev.off()

#' Participacion por sector en el consumo total, ordenada de forma ascendente.
#'
#' Se usa tanto en las barras horizontales como en la torta.
share <- aggregate(consumo_kwh ~ sector, data = df, FUN = sum)
share$pct <- share$consumo_kwh / sum(share$consumo_kwh) * 100
share <- share[order(share$pct), ]

#' Barras horizontales ordenadas: version CORRECTA de la torta.
#'
#' Compara longitudes sobre un eje comun de 0 a 100 % en lugar de angulos. La
#' primera pasada establece el area de trazado para colocar la cuadricula
#' vertical detras de las barras.
png(file.path(good_dir, "barh_sector_share.png"),
    width = 1950, height = 950, res = 300)
par(mar = c(4, 6, 3, 1))
bar_pos <- barplot(share$pct, names.arg = share$sector, horiz = TRUE,
                   col = NA, border = NA, xlim = c(0, 100),
                   main = "Participación por sector en el consumo total (R)",
                   xlab = "Participación (%)", las = 1)
grid(nx = NULL, ny = NA)
barplot(share$pct, horiz = TRUE, col = "#2b8cbe", border = NA,
        add = TRUE, axes = FALSE, names.arg = rep("", 3))
text(x = share$pct + 4, y = bar_pos,
     labels = sprintf("%.1f%%", share$pct), cex = 0.8)
dev.off()

#' GRAFICA MAL DISEÑADA (bad_design), ejemplo para analisis critico.
#'
#' Errores intencionales: titulo vago sin unidades, sin etiquetas de datos,
#' colores saturados sin funcion, leyenda separada, angulo de inicio arbitrario
#' y comparacion basada en angulos y areas.

#' Torta mal diseñada: version INCORRECTA.
png(file.path(bad_dir, "pie_sector_share_bad.png"),
    width = 1560, height = 1080, res = 300)
par(mar = c(1, 1, 3, 8))
pie(share$consumo_kwh, labels = NA,
    col = c("#ff00ff", "#00ffff", "#ffff00"),
    init.angle = 17, border = "grey40",
    main = "Consumo")
legend("topright", inset = c(-0.35, 0), xpd = TRUE,
       legend = share$sector, cex = 0.8,
       fill = c("#ff00ff", "#00ffff", "#ffff00"))
dev.off()

#' Verificacion de estadisticos: deben coincidir con Python, r esperado 0.998.
aggregate(consumo_kwh ~ sector, data = df, FUN = mean)
cor(df$consumo_kwh, df$costo_miles_cop)
