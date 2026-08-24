"""Paleta, tipografia y componentes reutilizables del dashboard.

Los colores son los de la identidad de la Universidad de La Salle (azul
institucional y dorado) y coinciden con los de ``utils/codes/python/estilo.py``,
de modo que una figura exportada para el informe y la misma figura vista en el
navegador usan exactamente el mismo color para el mismo grupo.

Aqui viven tambien los dos componentes que se repiten en toda la interfaz -la
tarjeta de indicador y el contenedor de tarjeta-, para que el modulo de layout
describa la estructura de la pagina y no sus estilos.
"""

from dash import html

# --- Barra lateral -----------------------------------------------------------
SIDEBAR_BG          = '#002D57'
SIDEBAR_ACTIVE_BG   = '#FFCD00'
SIDEBAR_TEXT        = '#FFFFFF'
SIDEBAR_MUTED       = '#7A99B8'

# --- Lienzo principal --------------------------------------------------------
MAIN_BG     = '#E6E6E6'
CARD_BG     = '#FFFFFF'
CARD_BORDER = '#E1E5EE'
SHADOW      = '0 2px 12px rgba(0,0,0,0.07)'

TITLE_COLOR = '#002D57'
TEXT_COLOR  = '#3D4A5C'
TEXT_MUTED  = '#94A3B8'

# --- Serie categorica (identica a estilo.py) ---------------------------------
GOLD      = '#FFCD00'
BLUE      = '#4472C4'
ORANGE    = '#ED7D31'
GREEN     = '#27AE60'
RED       = '#C0392B'
PURPLE    = '#8E44AD'
TURQUOISE = '#17A2B8'

# Se indexa por numero de cluster; el slider permite hasta k = 6.
CLUSTER_COLORS = [BLUE, ORANGE, GREEN, PURPLE, RED, TURQUOISE, GOLD]

SECTOR_COLORS = {
    'Residencial': BLUE,
    'Comercial': ORANGE,
    'Industrial': TITLE_COLOR,
}
SECTOR_ORDER = ['Residencial', 'Comercial', 'Industrial']

REGION_COLORS = {
    'Andina': GREEN,
    'Caribe': GOLD,
    'Pacifica': TURQUOISE,
}
REGION_ORDER = ['Andina', 'Caribe', 'Pacifica']

# Escala divergente para correlaciones, cargas y perfiles en z.
DIVERGING_SCALE = [
    [0.0, '#2166AC'], [0.25, '#89B5D8'], [0.5, '#F7F7F7'],
    [0.75, '#E28D77'], [1.0, '#B2182B'],
]

FONT = '"Segoe UI", Arial, sans-serif'

BASE_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_COLOR, family=FONT, size=12),
    margin=dict(l=60, r=30, t=58, b=50),
)

# Los desplegables se estilan en assets/dashboard.css y no aqui: en Dash 4 la
# prop ``style`` de dcc.Dropdown no llega al control como atributo inline, asi
# que un color de texto definido en Python acaba aplicandose sobre el fondo por
# defecto del componente. Ver el encabezado de esa hoja de estilos.
dd_container_style = {'marginBottom': '18px'}


def cluster_color(c):
    """Color asignado a un numero de cluster, ciclico si k supera la paleta."""
    return CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)]


def title_cfg(text, subtitle=None):
    """Configuracion de titulo para las figuras de Plotly del dashboard.

    Plotly no tiene subtitulo nativo: se incrusta como segunda linea del titulo
    con su propio tamano y color.
    """
    html_text = f'<b>{text}</b>'
    if subtitle:
        html_text += (
            f"<br><span style='font-size:11px;color:{TEXT_MUTED}'>{subtitle}</span>"
        )
    return dict(text=html_text, font=dict(size=14, color=TITLE_COLOR),
                x=0.012, xanchor='left', y=0.97, yanchor='top')


def kpi_card(icon, value, label, card_id=None, hint=None):
    """Tarjeta de indicador: icono, valor grande y etiqueta en versalitas.

    Parametros
    ----------
    icon : str
        Emoji que encabeza la tarjeta.
    value : str
        Valor inicial; los callbacks lo sobrescriben.
    label : str
        Descripcion corta del indicador.
    card_id : str, opcional
        ID del elemento del valor, para poder actualizarlo desde un callback.
    hint : str, opcional
        Aclaracion que aparece al pasar el cursor sobre la tarjeta.
    """
    return html.Div(
        [
            html.Div(icon, style={'fontSize': '24px', 'marginBottom': '8px'}),
            html.Div(value, id=card_id, style={
                'fontSize': '25px', 'fontWeight': '700',
                'color': TITLE_COLOR, 'lineHeight': '1',
            }),
            html.Div(label, style={
                'fontSize': '9.5px', 'color': TEXT_MUTED, 'marginTop': '6px',
                'textTransform': 'uppercase', 'letterSpacing': '0.8px',
                'fontWeight': '600',
            }),
        ],
        title=hint or '',
        style={
            'backgroundColor': CARD_BG,
            'border': f'1px solid {CARD_BORDER}',
            'padding': '18px 22px',
            'flex': '1',
            'boxShadow': SHADOW,
            'minWidth': '150px',
            'fontFamily': FONT,
        },
    )


def card(children, flex=1, padding=False, min_w='0'):
    """Contenedor blanco con borde y sombra: la unidad visual del lienzo."""
    style = {
        'flex': str(flex),
        'minWidth': min_w,
        'backgroundColor': CARD_BG,
        'border': f'1px solid {CARD_BORDER}',
        'boxShadow': SHADOW,
        'overflow': 'hidden',
    }
    if padding:
        style['padding'] = '18px 20px'
    return html.Div(children, style=style)


def section_label(text):
    """Rotulo de seccion en la barra lateral."""
    return html.Div(text, style={
        'color': SIDEBAR_MUTED, 'fontSize': '10px', 'fontWeight': '700',
        'letterSpacing': '1.4px', 'marginBottom': '14px',
    })


def field_label(text):
    """Etiqueta de un control de la barra lateral."""
    return html.Label(text, style={
        'color': SIDEBAR_TEXT, 'fontSize': '11px', 'fontWeight': '600',
        'display': 'block', 'marginBottom': '6px', 'letterSpacing': '0.3px',
    })
