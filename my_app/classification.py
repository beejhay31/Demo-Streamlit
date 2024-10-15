import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
#from my_app.model import Model

# Load Iris dataset
@st.cache_resource
def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return pd.DataFrame(X, columns=feature_names), pd.Series(y), target_names

# Train model
@st.cache_resource
def train_model(X, y):
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model
    """model = Model()
    reference_data = pd.concat([X, y.rename('target')], axis=1)
    model.train_model(reference_data, None)
    return model"""

# EDA and classification reporting
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("**Basic Statistics**")
    st.write(df.describe())

    st.write("**Class Distribution**")
    class_dist = pd.DataFrame(df['species'].value_counts()).reset_index()
    class_dist.columns = ['Species', 'Count']
    st.bar_chart(class_dist.set_index('Species'))

    st.write("**Pairplot**")
    pairplot_fig = sns.pairplot(df, hue='species')
    st.pyplot(pairplot_fig)

    st.write("**Correlation Heatmap**")
    numeric_df = df.drop(columns=['species'])
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("**Boxplots for Each Feature**")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=df, x='species', y='sepal length (cm)', ax=ax[0, 0])
    sns.boxplot(data=df, x='species', y='sepal width (cm)', ax=ax[0, 1])
    sns.boxplot(data=df, x='species', y='petal length (cm)', ax=ax[1, 0])
    sns.boxplot(data=df, x='species', y='petal width (cm)', ax=ax[1, 1])
    st.pyplot(fig)

# Model performance reporting
def model_performance(model, X_test, y_test, target_names):
    predictions = model.predict(X_test)
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    st.write("Classification Report:")
    st.text(classification_report(y_test, predictions, target_names=target_names))
    """# Use the performance_report method from the Model class in model.py
    current_data = pd.concat([X_test, y_test.rename('target')], axis=1)
    performance_report = model.performance_report(None, current_data)
    return performance_report"""
