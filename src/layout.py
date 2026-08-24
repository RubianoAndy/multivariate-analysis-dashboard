"""Estructura de la pagina: barra lateral de controles y lienzo de resultados.

El layout es estatico -describe que componentes existen y donde-, y todos los
valores llegan desde ``callbacks.py``. Las figuras se declaran vacias y se
rellenan en el primer disparo del callback, que Dash ejecuta al cargar.

La disposicion sigue el orden en que se lee un analisis multivariante:
indicadores, luego el plano de componentes (donde vive el resultado), luego las
piezas que lo explican (varianza, cargas, perfiles) y al final el detalle por
cliente.
"""

from dash import dcc, html, dash_table

from src.theme import (
    SIDEBAR_BG, SIDEBAR_TEXT, SIDEBAR_MUTED, GOLD,
    MAIN_BG, CARD_BG, CARD_BORDER, TITLE_COLOR, TEXT_COLOR, TEXT_MUTED, BLUE,
    FONT, dd_container_style,
    kpi_card, card, section_label, field_label,
)
from src.data import (
    LOGO_SRC, AUTHOR_SRC, sector_options, region_options, df, VARIABLES_MODELO,
)

SIDEBAR_WIDTH = '250px'

GRAPH_CONFIG = {'displaylogo': False, 'displayModeBar': False, 'responsive': True}
GRAPH_CONFIG_FULL = {'displaylogo': False, 'responsive': True,
                     'modeBarButtonsToRemove': ['select2d', 'lasso2d']}


sidebar = html.Div(
    [
        # --- Logo ------------------------------------------------------------
        html.Div(
            html.A(
                href='https://lasalle.edu.co/',
                target='_blank',
                rel='noopener noreferrer',
                style={'cursor': 'pointer', 'display': 'block'},
                children=html.Img(src=LOGO_SRC, style={
                    'width': '82%', 'maxWidth': '168px',
                    'display': 'block', 'margin': '0 auto',
                }),
            ) if LOGO_SRC else html.Div([
                html.P('UNIVERSIDAD', style={
                    'color': SIDEBAR_TEXT, 'fontSize': '11px', 'margin': '0',
                    'fontWeight': '700', 'letterSpacing': '1px', 'textAlign': 'center'}),
                html.P('DE LA SALLE', style={
                    'color': GOLD, 'fontSize': '13px', 'margin': '2px 0 0 0',
                    'fontWeight': '900', 'letterSpacing': '1px', 'textAlign': 'center'}),
            ]),
            style={
                'padding': '20px 16px 18px',
                'borderBottom': '1px solid rgba(255,255,255,0.1)',
                'flexShrink': '0',
            },
        ),

        # --- Controles -------------------------------------------------------
        html.Div(
            [
                section_label('FILTROS'),

                html.Div([
                    field_label('🏭  Sector'),
                    dcc.Dropdown(
                        id='filter-sector',
                        options=sector_options,
                        value=[],
                        multi=True,
                        placeholder='Todos los sectores',
                        clearable=True,
                        className='sidebar-dropdown',
                    ),
                ], style=dd_container_style),

                html.Div([
                    field_label('🗺️  Region'),
                    dcc.Dropdown(
                        id='filter-region',
                        options=region_options,
                        value=[],
                        multi=True,
                        placeholder='Todas las regiones',
                        clearable=True,
                        className='sidebar-dropdown',
                    ),
                ], style=dd_container_style),

                section_label('MODELO'),

                html.Div([
                    field_label('🎯  Numero de grupos (k)'),
                    dcc.Slider(
                        id='filter-k',
                        min=2, max=6, step=1, value=4,
                        marks={i: {'label': str(i),
                                   'style': {'color': SIDEBAR_MUTED, 'fontSize': '11px'}}
                               for i in range(2, 7)},
                        tooltip={'placement': 'bottom', 'always_visible': False},
                    ),
                    html.Div(
                        'El analisis se recalcula por completo sobre los clientes '
                        'filtrados: no se ocultan puntos, se vuelve a estimar el '
                        'PCA y a agrupar.',
                        style={'color': SIDEBAR_MUTED, 'fontSize': '10px',
                               'lineHeight': '1.45', 'marginTop': '10px'},
                    ),
                ], style={'marginBottom': '18px'}),

                html.Div([
                    field_label('🎨  Colorear el plano por'),
                    dcc.RadioItems(
                        id='filter-color',
                        options=[
                            {'label': ' Cluster descubierto', 'value': 'cluster'},
                            {'label': ' Sector declarado', 'value': 'sector'},
                            {'label': ' Region', 'value': 'region'},
                        ],
                        value='cluster',
                        labelStyle={'display': 'block', 'color': SIDEBAR_TEXT,
                                    'fontSize': '11.5px', 'marginBottom': '5px',
                                    'cursor': 'pointer'},
                        inputStyle={'marginRight': '6px'},
                    ),
                ], style=dd_container_style),

                html.Button(
                    '✕  Limpiar filtros',
                    id='btn-clear',
                    n_clicks=0,
                    style={
                        'width': '100%', 'padding': '8px 0',
                        'backgroundColor': 'transparent', 'color': SIDEBAR_MUTED,
                        'border': '1px solid rgba(255,255,255,0.12)',
                        'borderRadius': '6px', 'cursor': 'pointer',
                        'fontSize': '11px', 'fontWeight': '600',
                        'letterSpacing': '0.4px', 'marginTop': '4px',
                        'fontFamily': FONT, 'transition': 'all 0.2s',
                    },
                ),
            ],
            style={
                'padding': '20px 18px',
                'borderBottom': '1px solid rgba(255,255,255,0.1)',
                'flexShrink': '0',
            },
        ),

        # --- Nota metodologica -----------------------------------------------
        html.Div(
            [
                section_label('NOTA'),
                html.Div(
                    [
                        html.Span('El modelo usa '),
                        html.Span(f'{len(VARIABLES_MODELO)} variables',
                                  style={'color': SIDEBAR_TEXT, 'fontWeight': '600'}),
                        html.Span(' de las diez del conjunto: consumo, factor '
                                  'de potencia y antiguedad. Seis se descartaron '
                                  'por redundancia (correlacion sobre 0.90 entre '
                                  'ellas) y la temperatura por su indice KMO de '
                                  '0.456, bajo el umbral de 0.50.'),
                    ],
                    style={'color': SIDEBAR_MUTED, 'fontSize': '10.5px',
                           'lineHeight': '1.5'},
                ),
            ],
            style={'padding': '18px', 'borderBottom': '1px solid rgba(255,255,255,0.1)',
                   'flexShrink': '0'},
        ),

        # --- Autor -----------------------------------------------------------
        html.Div(
            html.Div(
                [
                    html.Img(src=AUTHOR_SRC, style={
                        'width': '46px', 'height': '46px', 'borderRadius': '50%',
                        'objectFit': 'cover', 'border': f'2px solid {GOLD}',
                        'marginRight': '10px', 'flexShrink': '0',
                    }) if AUTHOR_SRC else html.Div(style={
                        'width': '46px', 'height': '46px', 'borderRadius': '50%',
                        'backgroundColor': GOLD, 'marginRight': '10px',
                        'flexShrink': '0',
                    }),
                    html.Div([
                        html.P('Andy Rubiano', style={
                            'color': SIDEBAR_TEXT, 'margin': '0',
                            'fontSize': '13px', 'fontWeight': '600'}),
                        html.P('Analisis multivariante', style={
                            'color': SIDEBAR_MUTED, 'margin': '0', 'fontSize': '11px'}),
                    ]),
                ],
                style={'display': 'flex', 'alignItems': 'center'},
            ),
            style={
                'padding': '14px 18px', 'borderTop': '1px solid rgba(255,255,255,0.1)',
                'marginTop': 'auto', 'flexShrink': '0',
            },
        ),
    ],
    style={
        'width': SIDEBAR_WIDTH, 'minWidth': SIDEBAR_WIDTH,
        'backgroundColor': SIDEBAR_BG, 'display': 'flex', 'flexDirection': 'column',
        'height': '100vh', 'position': 'fixed', 'top': '0', 'left': '0',
        'zIndex': '100', 'overflowY': 'auto', 'overflowX': 'hidden',
        'fontFamily': FONT, 'boxSizing': 'border-box',
    },
)


encabezado = html.Div(
    [
        html.H1('ANALISIS MULTIVARIANTE DEL CONSUMO ELECTRICO', style={
            'margin': '0', 'fontSize': '21px', 'fontWeight': '800',
            'color': TITLE_COLOR, 'letterSpacing': '0.7px',
        }),
        html.P(
            f'Componentes principales y segmentacion de {len(df)} clientes  ·  '
            'Por Andy Rubiano  ·  Universidad de La Salle',
            style={'color': TEXT_MUTED, 'margin': '5px 0 0 0', 'fontSize': '12.5px'},
        ),
    ],
    style={
        'backgroundColor': CARD_BG, 'padding': '20px 30px',
        'borderBottom': f'1px solid {CARD_BORDER}',
        'boxShadow': '0 1px 4px rgba(0,0,0,0.05)', 'fontFamily': FONT,
    },
)


aviso_muestra = html.Div(
    id='aviso-muestra',
    style={'display': 'none'},
)


tabla_perfiles = dash_table.DataTable(
    id='table-perfiles',
    columns=[
        {'name': 'Grupo', 'id': 'Grupo'},
        {'name': 'Clientes', 'id': 'Clientes', 'type': 'numeric'},
        {'name': 'Consumo medio (kWh)', 'id': 'Consumo', 'type': 'numeric'},
        {'name': 'Potencia (kW)', 'id': 'Potencia', 'type': 'numeric'},
        {'name': 'F. potencia', 'id': 'FactorPotencia', 'type': 'numeric'},
        {'name': 'Antiguedad (anios)', 'id': 'Antiguedad', 'type': 'numeric'},
        {'name': 'Interrup./mes', 'id': 'Interrupciones', 'type': 'numeric'},
        {'name': 'Silueta', 'id': 'Silueta', 'type': 'numeric'},
    ],
    style_header={
        'backgroundColor': '#F8FAFC', 'color': TITLE_COLOR, 'fontWeight': '700',
        'border': f'1px solid {CARD_BORDER}', 'fontSize': '10.5px',
        'padding': '10px 10px', 'fontFamily': FONT, 'whiteSpace': 'normal',
        'height': 'auto',
    },
    style_cell={
        'backgroundColor': CARD_BG, 'color': TEXT_COLOR,
        'border': f'1px solid {CARD_BORDER}', 'fontSize': '11px',
        'padding': '9px 10px', 'textAlign': 'right', 'fontFamily': FONT,
    },
    style_cell_conditional=[
        {'if': {'column_id': 'Grupo'}, 'textAlign': 'left', 'minWidth': '210px',
         'fontWeight': '600'},
    ],
    style_data_conditional=[
        {'if': {'row_index': 'odd'}, 'backgroundColor': '#F8FAFC'},
        {'if': {'column_id': 'Clientes'}, 'fontWeight': '700', 'color': BLUE},
    ],
    style_table={'overflowX': 'auto'},
)


lienzo = html.Div(
    [
        aviso_muestra,

        # --- Indicadores -----------------------------------------------------
        html.Div(
            [
                kpi_card('👥', '', 'Clientes analizados', card_id='kpi-clientes',
                         hint='Numero de clientes que quedan tras aplicar los filtros'),
                kpi_card('🧭', '', 'Varianza en 2 componentes', card_id='kpi-varianza',
                         hint='Porcentaje de la informacion original que conservan PC1 y PC2'),
                kpi_card('🎯', '', 'Grupos formados', card_id='kpi-k',
                         hint='Valor de k seleccionado en el deslizador'),
                kpi_card('📐', '', 'Silueta media', card_id='kpi-silueta',
                         hint='Cohesion de la particion: por encima de 0.50 la '
                              'estructura es solida; entre 0.25 y 0.50, razonable'),
                kpi_card('⚡', '', 'Consumo agregado', card_id='kpi-consumo',
                         hint='Suma del consumo mensual de los clientes filtrados'),
            ],
            style={'display': 'flex', 'gap': '14px', 'marginBottom': '18px',
                   'flexWrap': 'wrap'},
        ),

        # --- Plano principal y varianza --------------------------------------
        html.Div(
            [
                card(dcc.Graph(id='fig-biplot', config=GRAPH_CONFIG_FULL,
                               style={'height': '520px'}), flex=62, min_w='420px'),
                card(dcc.Graph(id='fig-scree', config=GRAPH_CONFIG,
                               style={'height': '520px'}), flex=38, min_w='300px'),
            ],
            style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px',
                   'flexWrap': 'wrap'},
        ),

        # --- Cargas y perfiles -----------------------------------------------
        html.Div(
            [
                # La matriz de cargas tiene solo tres filas: sin bajarle la
                # altura sus celdas quedan enormes al lado del perfil.
                card(dcc.Graph(id='fig-cargas', config=GRAPH_CONFIG,
                               style={'height': '350px'}), flex=34, min_w='300px'),
                card(dcc.Graph(id='fig-perfil', config=GRAPH_CONFIG,
                               style={'height': '350px'}), flex=66, min_w='440px'),
            ],
            style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px',
                   'flexWrap': 'wrap'},
        ),

        # --- Coordenadas paralelas -------------------------------------------
        card(dcc.Graph(id='fig-paralelas', config=GRAPH_CONFIG_FULL,
                       style={'height': '410px'})),
        html.Div(style={'marginBottom': '16px'}),

        # --- Composicion y correlaciones -------------------------------------
        html.Div(
            [
                card(dcc.Graph(id='fig-sunburst', config=GRAPH_CONFIG,
                               style={'height': '470px'}), flex=42, min_w='320px'),
                card(dcc.Graph(id='fig-correlacion', config=GRAPH_CONFIG,
                               style={'height': '470px'}), flex=58, min_w='400px'),
            ],
            style={'display': 'flex', 'gap': '16px', 'marginBottom': '16px',
                   'flexWrap': 'wrap'},
        ),

        # --- Tabla de perfiles -----------------------------------------------
        card(
            [
                html.H3('Perfil de los grupos descubiertos', style={
                    'color': TITLE_COLOR, 'fontSize': '14px', 'fontWeight': '700',
                    'margin': '0 0 4px 0', 'fontFamily': FONT,
                }),
                html.P(
                    'Medias en unidades originales. El nombre de cada grupo se '
                    'deriva de su perfil en puntuaciones z y no interviene en el '
                    'modelo.',
                    style={'color': TEXT_MUTED, 'fontSize': '11px',
                           'margin': '0 0 14px 0', 'fontFamily': FONT},
                ),
                tabla_perfiles,
            ],
            padding=True,
        ),

        # --- Pie -------------------------------------------------------------
        html.Div(
            html.P(
                'Datos simulados con semilla fija (n = 300)  ·  '
                'PCA y K-Means con scikit-learn, verificados en R con prcomp y kmeans  ·  '
                'Construido con Python, Dash y Plotly',
                style={'color': TEXT_MUTED, 'textAlign': 'center',
                       'fontSize': '11px', 'margin': '0', 'fontFamily': FONT},
            ),
            style={'borderTop': f'1px solid {CARD_BORDER}', 'padding': '16px 0 6px',
                   'marginTop': '18px'},
        ),
    ],
    style={'padding': '22px 30px'},
)


layout = html.Div(
    [
        sidebar,
        html.Div(
            [encabezado, lienzo],
            style={
                'marginLeft': SIDEBAR_WIDTH, 'backgroundColor': MAIN_BG,
                'minHeight': '100vh', 'fontFamily': FONT,
            },
        ),
    ],
    style={'fontFamily': FONT},
)
