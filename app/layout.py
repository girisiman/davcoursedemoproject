from dash import dcc, html

layout = html.Div([
    html.H1("Titanic Survival Dashboard"),

    html.Label("Select Feature:"),
    dcc.Dropdown(
        id="feature-dropdown",
        options=[{"label": col, "value": col} for col in ["age", "pclass", "sex"]],
        value="age"
    ),

    dcc.Graph(id="feature-histogram"),
    dcc.Graph(id="survival-rate-plot"),

    html.H3("Hypothesis Testing"),
    dcc.Dropdown(
        id="hypothesis-dropdown",
        options=[{"label": col, "value": col} for col in ["sex", "pclass"]],
        value="sex"
    ),
    html.Div(id="hypothesis-output"),

    html.H3("Model Performance"),
    html.Div(id="model-output"),
])
