import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix, roc_curve, auc

def create_histogram(df, feature):
    fig = px.histogram(
        df, 
        x=feature, 
        color='survived', 
        barmode='overlay', 
        nbins=20
    )
    fig.update_layout(title=f"Distribution of {feature} by Survival")
    return fig

def create_survival_rate_plot(df, feature):
    grouped = df.groupby(feature)['survived'].mean().reset_index()
    fig = px.bar(
        grouped, 
        x=feature, 
        y='survived', 
        title=f"Survival Rate by {feature}"
    )
    return fig

def create_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Died", "Survived"],
        y=["Died", "Survived"],
        colorscale="Blues",
        showscale=True
    )
    fig.update_layout(title="Confusion Matrix")
    return fig

def create_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    fig = px.area(
        x=fpr, y=tpr,
        title=f"ROC Curve (AUC={auc_score:.2f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate")
    )
    return fig
