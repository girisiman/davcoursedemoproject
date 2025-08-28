import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from scipy.stats import chi2_contingency

def load_and_clean_data():
    df = sns.load_dataset('titanic')
    df.dropna(subset=['age', 'embarked', 'sex', 'class', 'survived'], inplace=True)
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['class'] = df['class'].map({'Third': 3, 'Second': 2, 'First': 1})
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

def run_hypothesis_test(df, cat_col):
    contingency = pd.crosstab(df[cat_col], df['survived'])
    chi2, p, _, _ = chi2_contingency(contingency)
    decision = "Reject H₀" if p < 0.05 else "Fail to reject H₀"
    return {"p_value": p, "decision": decision}

def run_model(df):
    X = df[['sex', 'age', 'class', 'fare', 'embarked']]
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    return model, report
