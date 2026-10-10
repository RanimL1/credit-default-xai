"""
Démarre l'API FastAPI dans le même conteneur que l'application Streamlit (hébergement gratuit).

L'application appelle l'API par HTTP (http://127.0.0.1:8000), exactement comme si elle était hébergée
ailleurs. Si une API tourne déjà à l'adresse API_URL (par exemple lancée à la main en local, ou
hébergée en ligne), rien n'est démarré.
"""
import threading
import time

import streamlit as st

from core.api_client import api_disponible, url_api

PORT = 8000


@st.cache_resource(show_spinner=False)
def _lancer_serveur():
    import uvicorn

    from api.main import app          # charge les modèles (quelques secondes)

    serveur = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=serveur.run, daemon=True).start()
    return serveur


def demarrer_api_si_besoin(attente=120):
    """Renvoie (api_prete, message_erreur)."""
    if not url_api().startswith(("http://localhost", "http://127.0.0.1")):
        return api_disponible(), None          # API distante : rien à lancer
    if api_disponible():
        return True, None
    try:
        _lancer_serveur()
    except Exception as e:                      # modèles absents, version incompatible, etc.
        return False, f"{type(e).__name__}: {e}"
    fin = time.time() + attente
    while time.time() < fin:
        if api_disponible():
            return True, None
        time.sleep(1)
    return False, "L'API n'a pas répondu dans le délai imparti."
