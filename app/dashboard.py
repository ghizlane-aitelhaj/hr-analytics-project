import dash
from dash import dcc, html, Input, Output, State
from simulation_service import load_model, predict_risk, get_counterfactuals

app = dash.Dash(__name__)
server = app.server

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────

page_style = {
    "minHeight": "100vh",
    "background": "linear-gradient(135deg, #0f172a 0%, #1f2937 45%, #3b0d11 100%)",
    "color": "white",
    "fontFamily": "Arial, sans-serif",
    "padding": "40px",
}

container_style = {
    "maxWidth": "1180px",
    "margin": "0 auto",
    "padding": "28px",
    "backgroundColor": "rgba(255,255,255,0.05)",
    "borderRadius": "18px",
    "boxShadow": "0 18px 60px rgba(0, 0, 0, 0.25)",
    "backdropFilter": "blur(8px)",
}

section_style = {
    "padding": "18px",
    "backgroundColor": "rgba(15, 23, 42, 0.42)",
    "border": "1px solid rgba(255,255,255,0.08)",
    "borderRadius": "14px",
}

field_group_style = {"marginBottom": "14px"}
label_style = {"display": "block", "marginBottom": "6px", "fontWeight": "700", "color": "#f8fafc"}
help_style = {"display": "block", "marginTop": "5px", "color": "#cbd5e1", "fontSize": "12px", "lineHeight": "1.35"}
input_style = {"width": "100%", "borderRadius": "8px", "border": "1px solid rgba(255,255,255,0.14)", "backgroundColor": "rgba(255,255,255,0.94)", "color": "#0f172a"}
dropdown_style = {"width": "100%", "backgroundColor": "rgba(255,255,255,0.94)", "color": "#0f172a"}


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSANTS UI
# ─────────────────────────────────────────────────────────────────────────────

def field_block(label, description, control):
    return html.Div(
        style=field_group_style,
        children=[
            html.Label(label, style=label_style),
            control,
            html.Small(description, style=help_style),
        ],
    )


def dropdown_control(**kwargs):
    return dcc.Dropdown(className="dark-dropdown", style=dropdown_style, **kwargs)


def section(title, children):
    return html.Div(
        style=section_style,
        children=[html.H3(title, style={"marginBottom": "14px", "color": "#f8fafc"}), *children],
    )


def render_scenario_card(scenario: dict, index: int, original_risk: float) -> html.Div:
    """
    Génère une carte visuelle pour un scénario contrefactuel DiCE.
    Affiche : les changements proposés + le nouveau score de risque estimé.
    """
    new_risk = scenario.get("new_risk_proba")
    delta_risk = (new_risk - original_risk) * 100 if new_risk is not None else None

    card_header_color = "#4ade80" if (new_risk is not None and new_risk < 0.3) else "#fbbf24"

    # Badge de réduction du risque
    if delta_risk is not None:
        badge = html.Span(
            f"{delta_risk:+.1f}% de risque",
            style={
                "backgroundColor": "rgba(74, 222, 128, 0.15)" if delta_risk < 0 else "rgba(251, 191, 36, 0.15)",
                "color": "#4ade80" if delta_risk < 0 else "#fbbf24",
                "borderRadius": "6px",
                "padding": "2px 10px",
                "fontSize": "12px",
                "fontWeight": "700",
                "marginLeft": "10px",
            },
        )
        new_risk_label = html.Span(
            f"Risque estimé : {new_risk * 100:.1f}%",
            style={"color": card_header_color, "fontWeight": "700", "fontSize": "14px"},
        )
    else:
        badge = html.Span()
        new_risk_label = html.Span()

    # Liste des changements proposés
    change_items = [
        html.Li(
            change,
            style={
                "marginBottom": "6px",
                "color": "#f1f5f9",
                "fontSize": "14px",
                "listStyleType": "none",
                "paddingLeft": "0",
            },
        )
        for change in scenario.get("changes", [])
    ]

    return html.Div(
        style={
            "backgroundColor": "rgba(15, 23, 42, 0.6)",
            "border": f"1px solid {card_header_color}40",
            "borderLeft": f"4px solid {card_header_color}",
            "borderRadius": "10px",
            "padding": "14px 18px",
            "marginBottom": "12px",
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "marginBottom": "10px"},
                children=[
                    html.Span(
                        f"Scénario {index + 1}",
                        style={"fontWeight": "700", "color": "#f8fafc", "fontSize": "15px"},
                    ),
                    badge,
                ],
            ),
            html.Ul(change_items, style={"margin": "0 0 8px 0", "padding": "0"}),
            new_risk_label,
        ],
    )


def render_dice_panel(scenarios: list, original_risk: float, original_income: float) -> html.Div:
    """
    Panneau complet de recommandations prescriptives DiCE.
    Affiché uniquement si le risque est élevé (> 50%).
    """
    if not scenarios:
        return html.Div(
            style={
                "marginTop": "20px",
                "padding": "14px",
                "backgroundColor": "rgba(251, 191, 36, 0.08)",
                "border": "1px solid rgba(251, 191, 36, 0.25)",
                "borderRadius": "10px",
            },
            children=[
                html.P(
                    "⚠️ Aucun scénario de rétention n'a pu être généré automatiquement.",
                    style={"color": "#fbbf24", "margin": "0", "fontSize": "14px"},
                ),
                html.P(
                    "Envisagez une discussion directe avec le collaborateur sur ses attentes.",
                    style={"color": "#cbd5e1", "margin": "6px 0 0 0", "fontSize": "13px"},
                ),
            ],
        )

    scenario_cards = [
        render_scenario_card(s, i, original_risk)
        for i, s in enumerate(scenarios)
        if s.get("changes")
    ]

    return html.Div(
        style={"marginTop": "24px"},
        children=[
            html.Hr(style={"borderColor": "rgba(255,255,255,0.08)", "margin": "0 0 18px 0"}),
            html.Div(
                style={"display": "flex", "alignItems": "center", "marginBottom": "14px"},
                children=[
                    html.Span("💡", style={"fontSize": "22px", "marginRight": "10px"}),
                    html.H4(
                        "Recommandations prescriptives IA (DiCE)",
                        style={"color": "#4ade80", "margin": "0", "fontSize": "16px", "fontWeight": "700"},
                    ),
                ],
            ),
            html.P(
                "Ces scénarios sont générés par l'algorithme de contrefactuels DiCE. "
                "Ils indiquent les changements minimaux permettant de ramener ce collaborateur "
                "sous le seuil de risque d'attrition.",
                style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "16px", "lineHeight": "1.5"},
            ),
            *scenario_cards,
            html.P(
                "ℹ️ Ces recommandations sont des suggestions algorithmiques. "
                "Elles doivent être validées par le manager et les RH avant toute décision.",
                style={
                    "color": "#64748b",
                    "fontSize": "12px",
                    "marginTop": "10px",
                    "fontStyle": "italic",
                    "lineHeight": "1.4",
                },
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

app.layout = html.Div(
    style=page_style,
    children=[
        html.Div(
            style=container_style,
            children=[
                html.H1("Simulateur de risque d'attrition", style={"color": "#f8fafc", "marginBottom": "8px"}),
                html.P(
                    "Remplissez les informations du candidat. Le risque d'attrition est calculé par le modèle Random Forest "
                    "et les recommandations prescriptives sont générées par l'algorithme DiCE.",
                    style={"color": "#cbd5e1", "marginBottom": "18px"},
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
                        "gap": "18px",
                        "marginTop": "18px",
                    },
                    children=[
                        section(
                            "Identité",
                            [
                                field_block(
                                    "Âge",
                                    "Âge du candidat en années (18-70).",
                                    dcc.Input(id="age", type="number", value=35, min=18, max=70, style=input_style),
                                ),
                                field_block(
                                    "Genre",
                                    "Genre du candidat.",
                                    dropdown_control(
                                        id="gender",
                                        options=[
                                            {"label": "Homme", "value": "Male"},
                                            {"label": "Femme", "value": "Female"},
                                        ],
                                        value="Male",
                                    ),
                                ),
                                field_block(
                                    "Situation familiale",
                                    "État matrimonial.",
                                    dropdown_control(
                                        id="maritalstatus",
                                        options=[
                                            {"label": "Célibataire", "value": "Single"},
                                            {"label": "Marié(e)", "value": "Married"},
                                            {"label": "Divorcé(e)", "value": "Divorced"},
                                        ],
                                        value="Single",
                                    ),
                                ),
                                field_block(
                                    "Niveau d'études",
                                    "Niveau de formation (1 à 5).",
                                    dcc.Input(id="education", type="number", value=3, min=1, max=5, style=input_style),
                                ),
                            ],
                        ),
                        section(
                            "Poste et conditions",
                            [
                                field_block(
                                    "Département",
                                    "Département d'affectation.",
                                    dropdown_control(
                                        id="department",
                                        options=[
                                            {"label": "Ventes", "value": "Sales"},
                                            {"label": "Recherche & Développement", "value": "Research & Development"},
                                            {"label": "Ressources Humaines", "value": "Human Resources"},
                                        ],
                                        value="Research & Development",
                                    ),
                                ),
                                field_block(
                                    "Distance domicile-travail (km)",
                                    "Distance en kilomètres.",
                                    dcc.Input(id="distance", type="number", value=10, min=0, style=input_style),
                                ),
                                field_block(
                                    "Expérience professionnelle (années)",
                                    "Années d'expérience totale.",
                                    dcc.Input(id="total_working_years", type="number", value=5, min=0, style=input_style),
                                ),
                                field_block(
                                    "Déplacements professionnels",
                                    "Fréquence des trajets professionnels.",
                                    dropdown_control(
                                        id="businesstravel",
                                        options=[
                                            {"label": "Aucun déplacement", "value": "Non-Travel"},
                                            {"label": "Déplacements rares", "value": "Travel_Rarely"},
                                            {"label": "Déplacements fréquents", "value": "Travel_Frequently"},
                                        ],
                                        value="Non-Travel",
                                    ),
                                ),
                            ],
                        ),
                        section(
                            "Rémunération",
                            [
                                field_block(
                                    "Salaire mensuel (€)",
                                    "Revenu mensuel brut en euros.",
                                    dcc.Input(id="monthly_income", type="number", value=3000, min=0, style=input_style),
                                ),
                                field_block(
                                    "Heures supplémentaires",
                                    "L'employé effectue-t-il des heures supplémentaires ?",
                                    dropdown_control(
                                        id="overtime",
                                        options=[
                                            {"label": "Non", "value": "No"},
                                            {"label": "Oui", "value": "Yes"},
                                        ],
                                        value="No",
                                    ),
                                ),
                            ],
                        ),
                        section(
                            "Résultat & Analyse",
                            [
                                html.Div(
                                    id="result",
                                    style={"color": "#f8fafc", "minHeight": "120px"},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"marginTop": "18px", "display": "flex", "justifyContent": "flex-end"},
                    children=[
                        html.Button(
                            "Analyser le risque",
                            id="btn",
                            n_clicks=0,
                            style={
                                "minWidth": "240px",
                                "padding": "12px 18px",
                                "background": "#b91c1c",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "10px",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "fontSize": "16px",
                            },
                        ),
                    ],
                ),
                # Zone des recommandations DiCE (pleine largeur sous la grille)
                html.Div(id="dice-panel", style={"marginTop": "10px"}),
                html.Hr(style={"marginTop": "20px", "borderColor": "rgba(255,255,255,0.06)"}),
            ],
        )
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("result", "children"),
    Output("dice-panel", "children"),
    Input("btn", "n_clicks"),
    State("age", "value"),
    State("gender", "value"),
    State("maritalstatus", "value"),
    State("education", "value"),
    State("department", "value"),
    State("distance", "value"),
    State("total_working_years", "value"),
    State("monthly_income", "value"),
    State("businesstravel", "value"),
    State("overtime", "value"),
    prevent_initial_call=True,
)
def run_simulation(
    n_clicks,
    age, gender, maritalstatus, education,
    department, distance, total_working_years,
    monthly_income, businesstravel, overtime,
):
    data = {
        "AGE": age or 35,
        "GENDER": gender,
        "MARITALSTATUS": maritalstatus,
        "EDUCATION": education or 3,
        "DEPARTMENT": department,
        "DISTANCEFROMHOME": distance or 10,
        "TOTALWORKINGYEARS": total_working_years or 5,
        "MONTHLYINCOME": monthly_income or 3000,
        "BUSINESSTRAVEL": businesstravel,
        "OVERTIME": overtime,
    }

    try:
        res = predict_risk(model, data)
        risk_score = res.get("risk_score", 0)
        risk_percent = risk_score * 100
        current_df = res.get("df")

        # ── Couleur et niveau de risque ──────────────────────────────────────
        if risk_percent < 30:
            color, level, icon = "#4ade80", "TRÈS FAIBLE", "✓"
            verdict = f"Excellent ! Ce collaborateur présente un risque très faible de départ ({risk_percent:.1f}%)."
        elif risk_percent < 50:
            color, level, icon = "#fbbf24", "MODÉRÉ", "⚠️"
            verdict = f"Attention : risque modéré de départ ({risk_percent:.1f}%). À surveiller lors du prochain entretien."
        else:
            color, level, icon = "#f87171", "ÉLEVÉ", "⛔"
            verdict = f"Alerte : ce collaborateur risque de quitter l'entreprise ({risk_percent:.1f}%). Action recommandée."

        # ── Bloc score principal ─────────────────────────────────────────────
        result_block = html.Div([
            html.Div(
                f"{risk_percent:.1f}%",
                style={"fontSize": "52px", "fontWeight": "800", "color": color, "marginBottom": "4px"},
            ),
            html.Div(
                f"{icon} Risque {level}",
                style={"fontSize": "18px", "fontWeight": "700", "color": color, "marginBottom": "10px"},
            ),
            html.Div(verdict, style={"fontSize": "14px", "lineHeight": "1.5", "color": "#e2e8f0"}),
            html.Div(
                f"Classe prédite : {res.get('prediction', 'N/A')}",
                style={"marginTop": "10px", "color": "#94a3b8", "fontSize": "12px"},
            ),
        ])

        # ── Panneau DiCE (uniquement si risque > 50%) ────────────────────────
        if risk_percent >= 50:
            scenarios = get_counterfactuals(model, current_df, n_cfs=3)
            dice_panel = render_dice_panel(scenarios, risk_score, float(current_df["MONTHLYINCOME"].iloc[0]))
        else:
            dice_panel = html.Div()

        return result_block, dice_panel

    except Exception as e:
        error_block = html.Div(
            f"Erreur lors de la prédiction : {str(e)}",
            style={"color": "#fca5a5", "fontSize": "14px"},
        )
        return error_block, html.Div()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)