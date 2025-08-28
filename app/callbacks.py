from dash import Output, Input
from .data_processing import load_and_clean_data, run_hypothesis_test, run_model
from .visualization import create_histogram, create_survival_rate_plot

df = load_and_clean_data()
model, report = run_model(df)

def register_callbacks(app):
    @app.callback(
        Output('feature-histogram', 'figure'),
        Input('feature-dropdown', 'value')
    )
    def update_histogram(feature):
        return create_histogram(df, feature)

    @app.callback(
        Output('survival-rate-plot', 'figure'),
        Input('feature-dropdown', 'value')
    )
    def update_survival_rate(feature):
        return create_survival_rate_plot(df, feature)

    @app.callback(
        Output("hypothesis-output", "children"),
        Input("hypothesis-dropdown", "value")
    )
    def update_hypothesis(cat_col):
        result = run_hypothesis_test(df, cat_col)
        return f"P-Value: {result['p_value']:.4f} → Decision: {result['decision']}"

    @app.callback(
        Output("model-output", "children"),
        Input("feature-dropdown", "value")
    )
    def update_model_results(_):
        acc = report['accuracy']
        precision = report['1']['precision']
        recall = report['1']['recall']
        return (f"Accuracy: {acc:.2f} | "
                f"Precision (Survived=1): {precision:.2f} | "
                f"Recall (Survived=1): {recall:.2f}")
