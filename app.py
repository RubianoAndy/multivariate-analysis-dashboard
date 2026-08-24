"""Punto de entrada del dashboard interactivo.

Levanta el servidor de desarrollo de Dash en http://localhost:8050/

    python app.py

Los datos deben existir antes de arrancar: el dashboard lee
``data/dataset/consumo_energia.csv`` y ``data/processed/resumen_modelo.csv``,
que producen las fases 0 y 2 del pipeline de ``utils/codes/python``.
"""

from src.dashboard import app, server

__all__ = ['app', 'server']

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
