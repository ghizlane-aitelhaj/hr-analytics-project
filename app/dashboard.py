import dash
from dash import dcc, html, Input, Output, State
from simulation_service import load_model, predict_risk, get_counterfactuals
from auth import verify_user, has_permission, get_demo_users, ROLES

app = dash.Dash(__name__, suppress_callback_exceptions=True, title="HR Analytics")
server = app.server
model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

page_style = {
    "minHeight": "100vh",
    "background": "linear-gradient(135deg, #0f172a 0%, #1f2937 45%, #3b0d11 100%)",
    "color": "white", "fontFamily": "Arial, sans-serif",
}
card_style = {
    "backgroundColor": "rgba(255,255,255,0.05)", "borderRadius": "18px",
    "boxShadow": "0 18px 60px rgba(0,0,0,0.25)", "backdropFilter": "blur(8px)",
    "border": "1px solid rgba(255,255,255,0.08)",
}
section_style = {
    "padding": "18px", "backgroundColor": "rgba(15,23,42,0.42)",
    "border": "1px solid rgba(255,255,255,0.08)", "borderRadius": "14px",
}
input_style = {
    "width": "100%", "borderRadius": "8px", "border": "1px solid rgba(255,255,255,0.14)",
    "backgroundColor": "rgba(255,255,255,0.94)", "color": "#0f172a",
    "padding": "8px 12px", "fontSize": "14px", "boxSizing": "border-box",
}
dropdown_style = {"width": "100%", "backgroundColor": "rgba(255,255,255,0.94)", "color": "#0f172a"}
label_style = {"display": "block", "marginBottom": "6px", "fontWeight": "700", "color": "#f8fafc", "fontSize": "13px"}
help_style = {"display": "block", "marginTop": "5px", "color": "#cbd5e1", "fontSize": "12px"}


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANTS RÉUTILISABLES
# ─────────────────────────────────────────────────────────────────────────────

def field_block(label, description, control):
    return html.Div(style={"marginBottom": "14px"}, children=[
        html.Label(label, style=label_style), control,
        html.Small(description, style=help_style),
    ])


def dropdown_control(**kwargs):
    return dcc.Dropdown(className="dark-dropdown", style=dropdown_style, **kwargs)


def section(title, children):
    return html.Div(style=section_style, children=[
        html.H3(title, style={"marginBottom": "14px", "color": "#f8fafc", "fontSize": "15px"}),
        *children,
    ])


def render_scenario_card(scenario, index, original_risk):
    new_risk = scenario.get("new_risk_proba")
    delta_risk = (new_risk - original_risk) * 100 if new_risk is not None else None
    card_color = "#4ade80" if (new_risk is not None and new_risk < 0.3) else "#fbbf24"
    badge = html.Span(
        f"{delta_risk:+.1f}% de risque",
        style={
            "backgroundColor": "rgba(74,222,128,0.15)" if (delta_risk and delta_risk < 0) else "rgba(251,191,36,0.15)",
            "color": "#4ade80" if (delta_risk and delta_risk < 0) else "#fbbf24",
            "borderRadius": "6px", "padding": "2px 10px", "fontSize": "12px",
            "fontWeight": "700", "marginLeft": "10px",
        },
    ) if delta_risk is not None else html.Span()

    return html.Div(
        style={
            "backgroundColor": "rgba(15,23,42,0.6)",
            "border": f"1px solid {card_color}40", "borderLeft": f"4px solid {card_color}",
            "borderRadius": "10px", "padding": "14px 18px", "marginBottom": "12px",
        },
        children=[
            html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}, children=[
                html.Span(f"Scénario {index + 1}", style={"fontWeight": "700", "color": "#f8fafc", "fontSize": "15px"}),
                badge,
            ]),
            html.Ul(
                [html.Li(c, style={"marginBottom": "6px", "color": "#f1f5f9", "fontSize": "14px",
                                   "listStyleType": "none", "paddingLeft": "0"})
                 for c in scenario.get("changes", [])],
                style={"margin": "0 0 8px 0", "padding": "0"},
            ),
            html.Span(f"Risque estimé : {new_risk * 100:.1f}%" if new_risk else "",
                      style={"color": card_color, "fontWeight": "700", "fontSize": "14px"}),
        ],
    )


def render_dice_panel(scenarios, original_risk, original_income):
    if not scenarios:
        return html.Div(
            style={"marginTop": "20px", "padding": "14px", "backgroundColor": "rgba(251,191,36,0.08)",
                   "border": "1px solid rgba(251,191,36,0.25)", "borderRadius": "10px"},
            children=[html.P("⚠️ Aucun scénario de rétention généré.", style={"color": "#fbbf24", "margin": "0"})],
        )
    cards = [render_scenario_card(s, i, original_risk) for i, s in enumerate(scenarios) if s.get("changes")]
    return html.Div(style={"marginTop": "24px"}, children=[
        html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "0 0 18px 0"}),
        html.Div(style={"display": "flex", "alignItems": "center", "marginBottom": "14px"}, children=[
            html.Span("💡", style={"fontSize": "22px", "marginRight": "10px"}),
            html.H4("Recommandations prescriptives IA (DiCE)",
                    style={"color": "#4ade80", "margin": "0", "fontSize": "16px", "fontWeight": "700"}),
        ]),
        html.P("Ces scénarios indiquent les changements minimaux pour ramener ce collaborateur sous le seuil de risque.",
               style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "16px"}),
        *cards,
        html.P("ℹ️ Suggestions algorithmiques à valider par le manager et les RH avant toute décision.",
               style={"color": "#64748b", "fontSize": "12px", "fontStyle": "italic", "marginTop": "10px"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# PAGE LOGIN
# ─────────────────────────────────────────────────────────────────────────────

def build_login_page(error_msg=""):
    demo_users = get_demo_users()
    demo_cards = []
    for u in demo_users:
        ri = ROLES.get(u["role"], {})
        demo_cards.append(html.Div(
            style={
                "backgroundColor": "rgba(255,255,255,0.04)",
                "border": f"1px solid {ri.get('color','#fff')}30",
                "borderLeft": f"3px solid {ri.get('color','#fff')}",
                "borderRadius": "8px", "padding": "10px 14px", "marginBottom": "8px",
            },
            children=[
                html.Div(style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                    html.Span(f"{ri.get('icon','')} {u['display_name']}",
                              style={"fontWeight": "700", "color": "#f8fafc", "fontSize": "13px"}),
                    html.Span(u["role"], style={
                        "backgroundColor": f"{ri.get('color','#fff')}20",
                        "color": ri.get("color", "#fff"),
                        "borderRadius": "4px", "padding": "2px 8px", "fontSize": "11px", "fontWeight": "700",
                    }),
                ]),
                html.Div(f"Login : {u['username']}  ·  Mot de passe : voir .env",
                         style={"color": "#94a3b8", "fontSize": "11px", "marginTop": "4px", "fontFamily": "monospace"}),
            ],
        ))

    return html.Div(
        style={**page_style, "display": "flex", "alignItems": "center", "justifyContent": "center", "padding": "40px"},
        children=[html.Div(style={**card_style, "width": "100%", "maxWidth": "480px", "padding": "40px"}, children=[
            html.Div(style={"textAlign": "center", "marginBottom": "32px"}, children=[
                html.Div("🏢", style={"fontSize": "48px", "marginBottom": "12px"}),
                html.H1("HR Analytics", style={"color": "#f8fafc", "fontSize": "24px", "fontWeight": "800", "margin": "0 0 6px 0"}),
                html.P("Authentification Active Directory", style={"color": "#94a3b8", "fontSize": "13px", "margin": "0"}),
            ]),
            html.Div(
                style={"backgroundColor": "rgba(96,165,250,0.1)", "border": "1px solid rgba(96,165,250,0.3)",
                       "borderRadius": "8px", "padding": "10px 14px", "marginBottom": "24px",
                       "display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Span("🔒", style={"fontSize": "16px"}),
                    html.Div([
                        html.Div("Connexion sécurisée via Active Directory",
                                 style={"color": "#60a5fa", "fontSize": "12px", "fontWeight": "700"}),
                        html.Div("entreprise.local · Domaine Windows Server 2022",
                                 style={"color": "#64748b", "fontSize": "11px"}),
                    ]),
                ],
            ),
            html.Div(style={"marginBottom": "16px"}, children=[
                html.Label("Identifiant Windows", style={**label_style, "marginBottom": "8px"}),
                dcc.Input(id="login-username", type="text", placeholder="ex: manager1",
                          style={**input_style, "marginBottom": "12px"}, debounce=False),
                html.Label("Mot de passe", style={**label_style, "marginBottom": "8px"}),
                dcc.Input(id="login-password", type="password", placeholder="Mot de passe",
                          style=input_style, debounce=False),
            ]),
            html.Div(error_msg, id="login-error", style={
                "color": "#f87171", "fontSize": "13px", "marginBottom": "12px",
                "display": "block" if error_msg else "none",
                "backgroundColor": "rgba(248,113,113,0.1)",
                "border": "1px solid rgba(248,113,113,0.3)" if error_msg else "none",
                "borderRadius": "6px", "padding": "8px 12px" if error_msg else "0",
            }),
            html.Button("🔐  Se connecter", id="login-btn", n_clicks=0, style={
                "width": "100%", "padding": "12px",
                "background": "linear-gradient(135deg, #1d4ed8, #1e40af)",
                "color": "white", "border": "none", "borderRadius": "10px",
                "fontWeight": "700", "cursor": "pointer", "fontSize": "15px",
                "marginBottom": "28px", "boxShadow": "0 4px 20px rgba(29,78,216,0.4)",
            }),
            html.Div(style={"borderTop": "1px solid rgba(255,255,255,0.08)", "paddingTop": "20px"}, children=[
                html.P("Comptes de démonstration :",
                       style={"color": "#64748b", "fontSize": "12px", "marginBottom": "12px", "fontWeight": "700"}),
                *demo_cards,
                html.P("💡 En production, remplacé par l'annuaire AD de l'entreprise (ldap3 / Azure AD).",
                       style={"color": "#475569", "fontSize": "11px", "marginTop": "12px", "fontStyle": "italic"}),
            ]),
        ])]
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def build_dashboard_page(user):
    ri = user.get("role_info", {})
    return html.Div(style={**page_style, "padding": "30px"}, children=[
        html.Div(style={"maxWidth": "1180px", "margin": "0 auto"}, children=[

            # Navbar
            html.Div(style={**card_style, "padding": "14px 24px", "marginBottom": "24px",
                            "display": "flex", "justifyContent": "space-between", "alignItems": "center"}, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                    html.Span("🏢", style={"fontSize": "24px"}),
                    html.Div([
                        html.Div("HR Analytics", style={"fontWeight": "800", "color": "#f8fafc", "fontSize": "16px"}),
                        html.Div("Risque d'attrition & Simulations IA", style={"color": "#94a3b8", "fontSize": "12px"}),
                    ]),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
                    html.Div(style={
                        "backgroundColor": f"{ri.get('color','#fff')}15",
                        "border": f"1px solid {ri.get('color','#fff')}40",
                        "borderRadius": "8px", "padding": "6px 14px",
                        "display": "flex", "alignItems": "center", "gap": "8px",
                    }, children=[
                        html.Span(ri.get("icon", ""), style={"fontSize": "16px"}),
                        html.Div([
                            html.Div(user.get("display_name", ""),
                                     style={"fontWeight": "700", "color": "#f8fafc", "fontSize": "13px"}),
                            html.Div(ri.get("label", user.get("role", "")),
                                     style={"color": ri.get("color", "#94a3b8"), "fontSize": "11px"}),
                        ]),
                    ]),
                    # Bouton déconnexion — utilise dcc.Store pour clearer la session
                    html.Button(
                        "Déconnexion",
                        id="logout-btn",
                        n_clicks=0,
                        style={
                            "padding": "8px 16px",
                            "backgroundColor": "rgba(248,113,113,0.15)",
                            "color": "#f87171",
                            "border": "1px solid rgba(248,113,113,0.3)",
                            "borderRadius": "8px", "cursor": "pointer",
                            "fontSize": "12px", "fontWeight": "700",
                        },
                    ),
                ]),
            ]),

            # Formulaire
            html.Div(style={**card_style, "padding": "28px"}, children=[
                html.H1("Simulateur de risque d'attrition",
                        style={"color": "#f8fafc", "marginBottom": "8px", "fontSize": "22px"}),
                html.P("Remplissez le profil du collaborateur. Le score est calculé par Random Forest "
                       "et les recommandations prescriptives par DiCE.",
                       style={"color": "#cbd5e1", "marginBottom": "18px", "fontSize": "13px"}),

                html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(0,1fr))", "gap": "18px"}, children=[
                    section("Identité", [
                        field_block("Âge", "18-70 ans.",
                                    dcc.Input(id="age", type="number", value=35, min=18, max=70, style=input_style)),
                        field_block("Genre", "Genre du collaborateur.",
                                    dropdown_control(id="gender", value="Male",
                                                     options=[{"label": "Homme", "value": "Male"},
                                                              {"label": "Femme", "value": "Female"}])),
                        field_block("Situation familiale", "État matrimonial.",
                                    dropdown_control(id="maritalstatus", value="Single",
                                                     options=[{"label": "Célibataire", "value": "Single"},
                                                              {"label": "Marié(e)", "value": "Married"},
                                                              {"label": "Divorcé(e)", "value": "Divorced"}])),
                        field_block("Niveau d'études", "1 à 5.",
                                    dcc.Input(id="education", type="number", value=3, min=1, max=5, style=input_style)),
                    ]),
                    section("Poste et conditions", [
                        field_block("Département", "Département d'affectation.",
                                    dropdown_control(id="department", value="Research & Development",
                                                     options=[{"label": "Ventes", "value": "Sales"},
                                                              {"label": "R&D", "value": "Research & Development"},
                                                              {"label": "RH", "value": "Human Resources"}])),
                        field_block("Distance domicile-travail (km)", "Distance en kilomètres.",
                                    dcc.Input(id="distance", type="number", value=10, min=0, style=input_style)),
                        field_block("Expérience (années)", "Années d'expérience totale.",
                                    dcc.Input(id="total_working_years", type="number", value=5, min=0, style=input_style)),
                        field_block("Déplacements professionnels", "Fréquence des trajets.",
                                    dropdown_control(id="businesstravel", value="Non-Travel",
                                                     options=[{"label": "Aucun", "value": "Non-Travel"},
                                                              {"label": "Rares", "value": "Travel_Rarely"},
                                                              {"label": "Fréquents", "value": "Travel_Frequently"}])),
                    ]),
                    section("Rémunération", [
                        field_block("Salaire mensuel (€)", "Revenu mensuel brut.",
                                    dcc.Input(id="monthly_income", type="number", value=3000, min=0, style=input_style)),
                        field_block("Heures supplémentaires", "L'employé fait-il des heures sup ?",
                                    dropdown_control(id="overtime", value="No",
                                                     options=[{"label": "Non", "value": "No"},
                                                              {"label": "Oui", "value": "Yes"}])),
                    ]),
                    section("Résultat & Analyse", [
                        html.Div(id="result", style={"color": "#f8fafc", "minHeight": "120px"}),
                    ]),
                ]),

                html.Div(style={"marginTop": "18px", "display": "flex", "justifyContent": "flex-end"}, children=[
                    html.Button("Analyser le risque", id="btn", n_clicks=0, style={
                        "minWidth": "240px", "padding": "12px 18px", "background": "#b91c1c",
                        "color": "white", "border": "none", "borderRadius": "10px",
                        "fontWeight": "700", "cursor": "pointer", "fontSize": "16px",
                    }),
                ]),
                html.Div(id="dice-panel", style={"marginTop": "10px"}),
            ]),
        ]),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT RACINE
# Les composants fantômes permettent à Dash de résoudre les callbacks
# même quand login-btn/logout-btn ne sont pas encore dans le DOM.
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Store(id="session-store", storage_type="session"),
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
    # Composants fantômes — invisibles, permettent la résolution des callbacks
    html.Div(id="login-btn",      style={"display": "none"}),
    html.Div(id="logout-btn",     style={"display": "none"}),
    html.Div(id="login-username", style={"display": "none"}),
    html.Div(id="login-password", style={"display": "none"}),
])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK ROUTER — gère login / logout / session
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("page-content", "children"),
    Output("session-store", "data"),
    Input("url", "pathname"),
    Input("url", "search"),          # déclenché à chaque navigation
    State("session-store", "data"),
    prevent_initial_call=False,
)
def router(pathname, search, session):
    # Déconnexion via paramètre URL ?logout=1
    if search and "logout=1" in search:
        return build_login_page(), None

    if session:
        return build_dashboard_page(session), session

    return build_login_page(), None


@app.callback(
    Output("url", "search"),
    Input("logout-btn", "n_clicks"),
    prevent_initial_call=True,
)
def trigger_logout(n_clicks):
    if n_clicks:
        return "?logout=1"
    return ""


@app.callback(
    Output("session-store", "data", allow_duplicate=True),
    Output("page-content", "children", allow_duplicate=True),
    Input("login-btn", "n_clicks"),
    State("login-username", "value"),
    State("login-password", "value"),
    prevent_initial_call=True,
)
def handle_login(n_clicks, username, password):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not username or not password:
        return None, build_login_page("Veuillez remplir l'identifiant et le mot de passe.")
    user = verify_user(username, password)
    if user:
        return user, build_dashboard_page(user)
    return None, build_login_page("❌ Identifiant ou mot de passe incorrect.")


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("result", "children"),
    Output("dice-panel", "children"),
    Input("btn", "n_clicks"),
    State("age", "value"), State("gender", "value"), State("maritalstatus", "value"),
    State("education", "value"), State("department", "value"), State("distance", "value"),
    State("total_working_years", "value"), State("monthly_income", "value"),
    State("businesstravel", "value"), State("overtime", "value"),
    State("session-store", "data"),
    prevent_initial_call=True,
)
def run_simulation(n_clicks, age, gender, maritalstatus, education, department,
                   distance, total_working_years, monthly_income, businesstravel, overtime, session):
    if not session:
        return html.Div("Session expirée. Reconnectez-vous.", style={"color": "#f87171"}), html.Div()

    data = {
        "AGE": age or 35, "GENDER": gender, "MARITALSTATUS": maritalstatus,
        "EDUCATION": education or 3, "DEPARTMENT": department,
        "DISTANCEFROMHOME": distance or 10, "TOTALWORKINGYEARS": total_working_years or 5,
        "MONTHLYINCOME": monthly_income or 3000, "BUSINESSTRAVEL": businesstravel, "OVERTIME": overtime,
    }

    try:
        res = predict_risk(model, data)
        risk_score = res.get("risk_score", 0)
        risk_percent = risk_score * 100
        current_df = res.get("df")

        if risk_percent < 30:
            color, level, icon = "#4ade80", "TRÈS FAIBLE", "✓"
            verdict = f"Excellent ! Risque très faible ({risk_percent:.1f}%)."
        elif risk_percent < 50:
            color, level, icon = "#fbbf24", "MODÉRÉ", "⚠️"
            verdict = f"Risque modéré ({risk_percent:.1f}%). À surveiller."
        else:
            color, level, icon = "#f87171", "ÉLEVÉ", "⛔"
            verdict = f"Alerte : risque élevé de départ ({risk_percent:.1f}%). Action recommandée."

        result_block = html.Div([
            html.Div(f"{risk_percent:.1f}%",
                     style={"fontSize": "52px", "fontWeight": "800", "color": color, "marginBottom": "4px"}),
            html.Div(f"{icon} Risque {level}",
                     style={"fontSize": "18px", "fontWeight": "700", "color": color, "marginBottom": "10px"}),
            html.Div(verdict, style={"fontSize": "14px", "lineHeight": "1.5", "color": "#e2e8f0"}),
            html.Div(f"Classe prédite : {res.get('prediction', 'N/A')}",
                     style={"marginTop": "10px", "color": "#94a3b8", "fontSize": "12px"}),
            html.Div(f"Analysé par : {session.get('display_name','')} ({session.get('role','')})",
                     style={"marginTop": "6px", "color": "#64748b", "fontSize": "11px", "fontStyle": "italic"}),
        ])

        if risk_percent >= 50 and has_permission(session, "view_dice"):
            scenarios = get_counterfactuals(model, current_df, n_cfs=3)
            dice_panel = render_dice_panel(scenarios, risk_score, float(current_df["MONTHLYINCOME"].iloc[0]))
        elif risk_percent >= 50:
            dice_panel = html.Div(
                "🔒 Recommandations DiCE réservées aux rôles RH et Admin.",
                style={"color": "#94a3b8", "fontSize": "13px", "marginTop": "16px", "fontStyle": "italic"},
            )
        else:
            dice_panel = html.Div()

        return result_block, dice_panel

    except Exception as e:
        return html.Div(f"Erreur : {str(e)}", style={"color": "#fca5a5"}), html.Div()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)