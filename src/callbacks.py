"""Callbacks: rehacen el analisis y redibujan las seis figuras del lienzo.

Un unico callback produce todas las salidas. Podria haber uno por figura, pero
las seis dependen del mismo objeto -el resultado del PCA y del clustering sobre
el subconjunto filtrado-, y separarlas obligaria a repetir ese calculo seis
veces por interaccion o a coordinarlas con un almacen intermedio. Con 300
clientes el analisis completo tarda milisegundos, asi que la version simple es
tambien la rapida.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, html

from src.theme import (
    BASE_LAYOUT, title_cfg, cluster_color,
    TITLE_COLOR, TEXT_COLOR, TEXT_MUTED, CARD_BORDER, GOLD, RED,
    SECTOR_COLORS, SECTOR_ORDER, REGION_COLORS, REGION_ORDER,
    DIVERGING_SCALE, FONT,
)
from src.data import (
    analizar, MINIMO_CLIENTES, ETIQUETAS_CORTAS, VARIABLES_LOG,
    VARIABLES_NUMERICAS, eje_componente,
)

GRID = '#F1F3F8'

# Etiquetas cortas de todas las variables numericas, con el prefijo log donde
# corresponde. Se usa en las figuras que muestran la matriz completa.
def etiqueta(variable, con_log=True):
    base = ETIQUETAS_CORTAS.get(variable, variable)
    return f'log {base}' if (con_log and variable in VARIABLES_LOG) else base


def figura_vacia(mensaje):
    """Figura de sustitucion cuando el filtro deja una muestra insuficiente."""
    fig = go.Figure()
    fig.add_annotation(
        text=mensaje, showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5,
        font=dict(size=13, color=TEXT_MUTED, family=FONT),
    )
    fig.update_layout(**BASE_LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def color_por(datos, criterio):
    """Devuelve (columna, mapa de colores, orden) segun el criterio elegido."""
    if criterio == 'sector':
        return 'sector', SECTOR_COLORS, [s for s in SECTOR_ORDER
                                         if s in datos['sector'].unique()]
    if criterio == 'region':
        return 'region', REGION_COLORS, [r for r in REGION_ORDER
                                         if r in datos['region'].unique()]
    nombres = dict(zip(datos['cluster'], datos['nombre_cluster']))
    mapa = {nombres[c]: cluster_color(c) for c in sorted(nombres)}
    return 'nombre_cluster', mapa, [nombres[c] for c in sorted(nombres)]


# -----------------------------------------------------------------------------
# FIGURAS
# -----------------------------------------------------------------------------
def fig_biplot(resultado, criterio):
    """Plano PC1-PC2 con las cargas superpuestas y detalle por cliente."""
    datos = resultado['datos']
    columna, mapa, orden = color_por(datos, criterio)

    fig = px.scatter(
        datos, x='PC1', y='PC2', color=columna,
        color_discrete_map=mapa, category_orders={columna: orden},
        custom_data=['id_cliente', 'sector', 'region', 'consumo_kwh',
                     'factor_potencia', 'antiguedad_anios'],
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.78, line=dict(width=0.5, color='white')),
        hovertemplate=(
            '<b>%{customdata[0]}</b><br>%{customdata[1]} | %{customdata[2]}'
            '<br><br>Consumo: %{customdata[3]:,.0f} kWh/mes'
            '<br>Factor de potencia: %{customdata[4]:.3f}'
            '<br>Antiguedad: %{customdata[5]:.1f} anios'
            '<extra></extra>'
        ),
    )

    # Vectores de carga, escalados a la nube y con la etiqueta escalonada en
    # perpendicular: las seis variables de tamano son casi colineales y sus
    # etiquetas se apilarian una sobre otra.
    cargas = resultado['cargas']
    escala = 0.75 * max(datos['PC1'].abs().max(), datos['PC2'].abs().max())
    separacion = 0.26 * escala
    colocadas = []
    orden_cargas = cargas.reindex(
        cargas['PC1'].abs().add(cargas['PC2'].abs()).sort_values(ascending=False).index
    )

    for variable in orden_cargas.index:
        x = cargas.loc[variable, 'PC1'] * escala
        y = cargas.loc[variable, 'PC2'] * escala
        fig.add_annotation(x=x, y=y, ax=0, ay=0, xref='x', yref='y',
                           axref='x', ayref='y', text='', showarrow=True,
                           arrowhead=2, arrowsize=1, arrowwidth=1.6,
                           arrowcolor=TITLE_COLOR, opacity=0.85)

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
            x=destino[0], y=destino[1], text=etiqueta(variable.replace('log_', '')),
            showarrow=False, font=dict(size=10, color=TITLE_COLOR),
            bgcolor='rgba(255,255,255,0.85)', borderpad=2,
        )

    varianza = resultado['varianza']
    cargas_pc = resultado['cargas']
    fig.update_layout(
        # Margen inferior holgado: la leyenda va debajo del eje y con cuatro
        # grupos de nombre largo ocupa dos filas.
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=60, r=30, t=58, b=105),
        title=title_cfg(
            'Plano de las dos primeras componentes',
            'Cada punto es un cliente; las flechas indican hacia donde crece cada variable',
        ),
        xaxis_title=eje_componente(cargas_pc, varianza, 'PC1'),
        yaxis_title=eje_componente(cargas_pc, varianza, 'PC2'),
        legend=dict(title='', orientation='h', yanchor='top', y=-0.13,
                    x=0, font=dict(size=10.5)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=True, zerolinecolor=CARD_BORDER)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=CARD_BORDER)
    return fig


def fig_scree(resultado):
    """Varianza explicada por componente y acumulada, con el corte de Kaiser."""
    varianza = resultado['varianza']
    retenidas = int((varianza['criterio_varianza_acumulada'] == 'Retener').sum())

    fig = go.Figure()
    fig.add_bar(
        x=varianza['componente'], y=varianza['varianza_explicada_pct'],
        marker_color=[TITLE_COLOR if i < retenidas else '#B9C4D4'
                      for i in range(len(varianza))],
        name='Por componente',
        hovertemplate='%{x}: %{y:.2f} %<extra></extra>',
        text=[f'{v:.1f}%' if v >= 3 else '' for v in varianza['varianza_explicada_pct']],
        textposition='outside', textfont=dict(size=10, color=TEXT_COLOR),
    )
    fig.add_scatter(
        x=varianza['componente'], y=varianza['varianza_acumulada_pct'],
        mode='lines+markers', name='Acumulada', yaxis='y2',
        line=dict(color=GOLD, width=2.5), marker=dict(size=6),
        hovertemplate='%{x}: %{y:.2f} % acumulado<extra></extra>',
    )
    fig.add_hline(y=80, line=dict(color=RED, dash='dash', width=1.2), yref='y2')

    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=55, r=55, t=58, b=50),
        title=title_cfg(
            'Varianza explicada',
            f'Se retienen {retenidas} componentes: las que acumulan mas del 80 % '
            f'de la varianza',
        ),
        xaxis_title='Componente principal',
        yaxis=dict(title='Varianza explicada (%)', gridcolor=GRID),
        yaxis2=dict(title='Acumulada (%)', overlaying='y', side='right',
                    range=[0, 105], showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=-0.20, x=0,
                    font=dict(size=10.5)),
        bargap=0.35,
    )
    return fig


def fig_cargas(resultado):
    """Mapa de calor de las cargas de las componentes retenidas."""
    cargas = resultado['cargas']
    n = resultado['n_componentes']
    matriz = cargas.iloc[:, :n]
    etiquetas_y = [etiqueta(v.replace('log_', '')) for v in matriz.index]

    fig = go.Figure(
        go.Heatmap(
            z=matriz.values, x=list(matriz.columns), y=etiquetas_y,
            colorscale=DIVERGING_SCALE, zmid=0, zmin=-1, zmax=1,
            text=[[f'{v:+.2f}' for v in fila] for fila in matriz.values],
            texttemplate='%{text}', textfont=dict(size=10),
            hovertemplate='%{y} en %{x}: r = %{z:.3f}<extra></extra>',
            colorbar=dict(title=dict(text='r', side='top'), thickness=12, len=0.7),
            xgap=2, ygap=2,
        )
    )
    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=110, r=20, t=58, b=40),
        title=title_cfg(
            'Cargas de las componentes',
            'Correlacion entre cada variable original y cada componente',
        ),
    )
    fig.update_yaxes(autorange='reversed', tickfont=dict(size=10.5))
    return fig


def fig_perfil(resultado):
    """Perfil de cada grupo en puntuaciones z, incluida la variable excluida."""
    perfil_z = resultado['perfil_z']
    tamanos = resultado['datos']['cluster'].value_counts()

    columnas = list(VARIABLES_NUMERICAS)
    matriz = perfil_z[columnas]
    etiquetas_x = [ETIQUETAS_CORTAS.get(c, c) for c in columnas]
    etiquetas_y = [f'C{c} (n={tamanos.get(c, 0)})' for c in matriz.index]

    fig = go.Figure(
        go.Heatmap(
            z=matriz.values, x=etiquetas_x, y=etiquetas_y,
            colorscale=DIVERGING_SCALE, zmid=0, zmin=-1.5, zmax=1.5,
            text=[[f'{v:+.2f}' for v in fila] for fila in matriz.values],
            texttemplate='%{text}', textfont=dict(size=10),
            hovertemplate='%{y} en %{x}: %{z:+.2f} desviaciones<extra></extra>',
            colorbar=dict(title=dict(text='z', side='top'), thickness=12, len=0.7),
            xgap=2, ygap=2,
        )
    )
    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=95, r=20, t=58, b=80),
        title=title_cfg(
            'Perfil de cada grupo en puntuaciones z',
            'Desviaciones tipicas respecto a la media de los clientes mostrados',
        ),
    )
    fig.update_xaxes(tickangle=-32, tickfont=dict(size=10))
    fig.update_yaxes(autorange='reversed', tickfont=dict(size=10.5))
    return fig


def fig_paralelas(resultado):
    """Coordenadas paralelas de las variables del modelo, por grupo."""
    datos = resultado['datos']
    variables = resultado['variables']

    matriz = datos[variables].copy()
    for col in [c for c in VARIABLES_LOG if c in matriz.columns]:
        matriz[col] = np.log(matriz[col])
    desv = matriz.std().replace(0, np.nan)
    matriz = ((matriz - matriz.mean()) / desv).fillna(0).clip(-3, 3)

    clusters = sorted(datos['cluster'].unique())
    escala = []
    for i, c in enumerate(clusters):
        escala.append([i / len(clusters), cluster_color(c)])
        escala.append([(i + 1) / len(clusters), cluster_color(c)])

    fig = go.Figure(
        go.Parcoords(
            line=dict(
                color=datos['cluster'], colorscale=escala,
                cmin=-0.5, cmax=len(clusters) - 0.5, showscale=True,
                colorbar=dict(title=dict(text='Grupo', side='top'),
                              tickvals=clusters, ticktext=[f'C{c}' for c in clusters],
                              thickness=12, len=0.6),
            ),
            # Los ticks van en los enteros interiores: Plotly ya rotula los
            # extremos del rango y un tick en el mismo punto se imprime encima.
            dimensions=[
                dict(label=etiqueta(v), values=matriz[v], range=[-3, 3],
                     tickvals=[-2, -1, 0, 1, 2])
                for v in variables
            ],
            labelangle=-16,
            labelfont=dict(size=10.5, color=TITLE_COLOR),
            tickfont=dict(size=9),
        )
    )
    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=80, r=70, t=88, b=30),
        title=title_cfg(
            'Perfil de los clientes en coordenadas paralelas',
            'Variables en puntuaciones z. Arrastre un intervalo sobre cualquier eje para filtrar',
        ),
    )
    return fig


def fig_sunburst(resultado):
    """Composicion jerarquica grupo > sector > region, ponderada por consumo."""
    datos = resultado['datos'].copy()
    datos['Grupo'] = datos['cluster'].map(lambda c: f'C{c}')

    fig = px.sunburst(
        datos, path=['Grupo', 'sector', 'region'], values='consumo_kwh',
        color='Grupo',
        color_discrete_map={f'C{c}': cluster_color(c)
                            for c in sorted(datos['cluster'].unique())},
    )
    fig.update_traces(
        hovertemplate=('<b>%{label}</b><br>Consumo agregado: %{value:,.0f} kWh/mes'
                       '<br>%{percentRoot:.1%} del total mostrado<extra></extra>'),
        insidetextorientation='radial',
        marker=dict(line=dict(color='white', width=1.4)),
    )
    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=10, r=10, t=68, b=10),
        title=title_cfg(
            'Composicion de cada grupo',
            'El area es el consumo agregado, no el numero de clientes',
        ),
    )
    return fig


def fig_correlacion(resultado):
    """Matriz de correlacion de las variables del modelo, triangular inferior.

    Con tres variables la matriz cabe entera y se lee de un vistazo; el valor
    que importa es el -0.94 entre factor de potencia y antiguedad, que es la
    redundancia que el PCA comprime en una sola componente.
    """
    datos = resultado['datos']
    variables = resultado['variables']

    matriz = datos[variables].copy()
    for col in [c for c in VARIABLES_LOG if c in matriz.columns]:
        matriz[col] = np.log(matriz[col])
    R = matriz.corr()

    # Se oculta el triangulo superior: es simetrico y solo anade ruido visual.
    z = R.values.astype(float).copy()
    z[np.triu_indices_from(z, k=1)] = np.nan
    etiquetas = [etiqueta(v) for v in variables]

    fig = go.Figure(
        go.Heatmap(
            z=z, x=etiquetas, y=etiquetas,
            colorscale=DIVERGING_SCALE, zmid=0, zmin=-1, zmax=1,
            text=[[('' if np.isnan(v) else f'{v:.2f}') for v in fila] for fila in z],
            texttemplate='%{text}', textfont=dict(size=9),
            hovertemplate='%{y} vs %{x}: r = %{z:.3f}<extra></extra>',
            colorbar=dict(title=dict(text='r', side='top'), thickness=12, len=0.7),
            xgap=2, ygap=2, hoverongaps=False,
        )
    )
    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k != 'margin'},
        margin=dict(l=110, r=20, t=58, b=95),
        title=title_cfg(
            'Correlaciones entre las tres variables',
            'Factor de potencia y antiguedad describen la misma realidad: r = -0.94',
        ),
    )
    fig.update_xaxes(tickangle=-38, tickfont=dict(size=9.5))
    fig.update_yaxes(autorange='reversed', tickfont=dict(size=9.5))
    return fig


def tabla_perfiles(resultado):
    """Filas de la tabla de perfiles, en unidades originales."""
    perfil = resultado['perfil']
    datos = resultado['datos']
    silueta_grupo = datos.groupby('cluster')['silueta'].mean()

    filas = []
    for c in perfil.index:
        filas.append({
            'Grupo': resultado['nombres_cluster'][c],
            'Clientes': int(perfil.loc[c, 'n_clientes']),
            'Consumo': round(perfil.loc[c, 'consumo_kwh'], 1),
            'FactorPotencia': round(perfil.loc[c, 'factor_potencia'], 3),
            'Antiguedad': round(perfil.loc[c, 'antiguedad_anios'], 1),
            'Silueta': round(float(silueta_grupo.get(c, float('nan'))), 3),
        })
    return filas


# -----------------------------------------------------------------------------
def register_callbacks(app):
    """Registra los callbacks de la aplicacion."""

    @app.callback(
        Output('filter-sector', 'value'),
        Output('filter-region', 'value'),
        Output('filter-k', 'value'),
        Input('btn-clear', 'n_clicks'),
        prevent_initial_call=True,
    )
    def limpiar_filtros(_):
        """Devuelve los tres controles a su estado inicial."""
        return [], [], 4

    @app.callback(
        Output('kpi-clientes', 'children'),
        Output('kpi-varianza', 'children'),
        Output('kpi-k', 'children'),
        Output('kpi-silueta', 'children'),
        Output('kpi-consumo', 'children'),
        Output('aviso-muestra', 'children'),
        Output('aviso-muestra', 'style'),
        Output('fig-biplot', 'figure'),
        Output('fig-scree', 'figure'),
        Output('fig-cargas', 'figure'),
        Output('fig-perfil', 'figure'),
        Output('fig-paralelas', 'figure'),
        Output('fig-sunburst', 'figure'),
        Output('fig-correlacion', 'figure'),
        Output('table-perfiles', 'data'),
        Input('filter-sector', 'value'),
        Input('filter-region', 'value'),
        Input('filter-k', 'value'),
        Input('filter-color', 'value'),
    )
    def actualizar(sectores, regiones, k, criterio_color):
        """Recalcula el analisis y regenera indicadores, figuras y tabla.

        Parametros
        ----------
        sectores, regiones : list[str]
            Seleccion de los filtros; lista vacia significa "todos".
        k : int
            Numero de grupos pedido en el deslizador.
        criterio_color : {'cluster', 'sector', 'region'}
            Variable que colorea el plano principal. Permite contrastar la
            particion descubierta con las etiquetas que ya se conocian.

        Retorna
        -------
        tuple
            Cinco indicadores, el aviso de muestra y su estilo, siete figuras y
            las filas de la tabla, en el orden declarado en el decorador.
        """
        resultado = analizar(sectores, regiones, k)

        if resultado is None:
            mensaje = (
                f'La combinacion de filtros deja menos de {MINIMO_CLIENTES} '
                'clientes. Con esa muestra la matriz de correlacion es inestable '
                'y el analisis no seria fiable: amplie la seleccion.'
            )
            aviso = html.Div(
                mensaje,
                style={
                    'backgroundColor': '#FFF4E5', 'border': f'1px solid {GOLD}',
                    'color': '#7A5B00', 'padding': '14px 18px',
                    'fontSize': '12.5px', 'marginBottom': '18px',
                    'fontFamily': FONT,
                },
            )
            vacia = figura_vacia('Muestra insuficiente para el analisis')
            return ('—', '—', '—', '—', '—', aviso, {'display': 'block'},
                    vacia, vacia, vacia, vacia, vacia, vacia, vacia, [])

        datos = resultado['datos']
        varianza = resultado['varianza']
        var_2 = varianza.loc[1, 'varianza_acumulada_pct']
        consumo_total = datos['consumo_kwh'].sum() / 1000

        silueta = resultado['silueta']
        silueta_txt = '—' if np.isnan(silueta) else f'{silueta:.3f}'

        return (
            f'{len(datos):,}'.replace(',', '.'),
            f'{var_2:.1f} %',
            str(resultado['k']),
            silueta_txt,
            f'{consumo_total:,.0f} MWh'.replace(',', '.'),
            None,
            {'display': 'none'},
            fig_biplot(resultado, criterio_color),
            fig_scree(resultado),
            fig_cargas(resultado),
            fig_perfil(resultado),
            fig_paralelas(resultado),
            fig_sunburst(resultado),
            fig_correlacion(resultado),
            tabla_perfiles(resultado),
        )
