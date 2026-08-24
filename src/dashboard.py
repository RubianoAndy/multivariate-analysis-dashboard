"""Construccion de la aplicacion Dash: layout mas callbacks.

Se separa de ``app.py`` para que el objeto ``app`` pueda importarse desde un
servidor WSGI (gunicorn, waitress) sin ejecutar el bloque de arranque.
"""

from pathlib import Path

import dash

from src.layout import layout
from src.callbacks import register_callbacks

BASE_DIR = Path(__file__).resolve().parents[1]

# Dash busca la carpeta de recursos estaticos junto al modulo donde se crea la
# aplicacion, que aqui es src/. Como la hoja de estilos vive en la raiz del
# proyecto -junto a app.py, que es donde se espera encontrarla-, hay que
# indicarle la ruta de forma explicita; sin esto el CSS no se sirve y los
# controles se quedan con la apariencia por defecto de Dash.
app = dash.Dash(
    __name__,
    title='Analisis multivariante - Consumo electrico',
    update_title='Recalculando...',
    assets_folder=str(BASE_DIR / 'assets'),
    suppress_callback_exceptions=True,
)
server = app.server

app.layout = layout
register_callbacks(app)
