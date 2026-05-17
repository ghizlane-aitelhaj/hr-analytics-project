"""
auth.py — Simulation Active Directory pour HR Analytics Dashboard.

En production, verify_user() appellerait ldap3 :
    from ldap3 import Server, Connection, ALL
    server = Server(AD_SERVER, get_info=ALL)
    conn = Connection(server, user=f"{username}@{AD_DOMAIN}", password=password)
    return conn.bind()

Pour ce projet, les utilisateurs sont définis dans .env :
    AD_USERS=manager1:password1:Manager,rh1:password2:RH,admin:password3:Admin
"""

import os
import hashlib
from dotenv import load_dotenv

load_dotenv()

ROLES = {
    "Manager": {
        "label": "Manager",
        "icon": "👔",
        "color": "#60a5fa",
        "permissions": ["view_team", "view_risk", "view_dice"],
        "description": "Accès aux employés de son équipe et aux recommandations DiCE",
    },
    "RH": {
        "label": "Ressources Humaines",
        "icon": "🧑‍💼",
        "color": "#4ade80",
        "permissions": ["view_team", "view_risk", "view_dice", "view_all_employees"],
        "description": "Accès complet à tous les employés et aux simulations",
    },
    "Admin": {
        "label": "Administrateur",
        "icon": "⚙️",
        "color": "#f59e0b",
        "permissions": ["view_team", "view_risk", "view_dice", "view_all_employees", "view_mlflow"],
        "description": "Accès complet + métriques MLflow et gestion des modèles",
    },
}


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load_users() -> dict:
    users_env = os.getenv("AD_USERS", "")
    if users_env:
        users = {}
        for entry in users_env.split(","):
            parts = entry.strip().split(":")
            if len(parts) == 3:
                username, password, role = parts
                users[username.lower()] = {
                    "password_hash": _hash(password),
                    "role": role,
                    "display_name": username.capitalize(),
                    "email": f"{username.lower()}@entreprise.com",
                }
        return users

    # Utilisateurs de démo par défaut
    return {
        "manager1": {"password_hash": _hash("manager123"), "role": "Manager",
                     "display_name": "Sophie Martin", "email": "s.martin@entreprise.com"},
        "rh1":      {"password_hash": _hash("rh123"),      "role": "RH",
                     "display_name": "Karim Benali",  "email": "k.benali@entreprise.com"},
        "admin":    {"password_hash": _hash("admin123"),   "role": "Admin",
                     "display_name": "Admin Système",  "email": "admin@entreprise.com"},
    }


USERS = _load_users()


def verify_user(username: str, password: str) -> dict | None:
    """
    Vérifie les credentials et retourne le profil utilisateur ou None.
    En production → remplacer le corps par un appel ldap3.
    """
    user = USERS.get(username.lower())
    if user and user["password_hash"] == _hash(password):
        return {
            "username": username.lower(),
            "display_name": user["display_name"],
            "email": user["email"],
            "role": user["role"],
            "role_info": ROLES.get(user["role"], {}),
        }
    return None


def has_permission(session_user: dict | None, permission: str) -> bool:
    if not session_user:
        return False
    role = session_user.get("role", "")
    return permission in ROLES.get(role, {}).get("permissions", [])


def get_demo_users() -> list[dict]:
    return [
        {"username": u, "role": d["role"], "display_name": d["display_name"]}
        for u, d in USERS.items()
    ]
