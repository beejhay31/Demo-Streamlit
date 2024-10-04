import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Placeholder for user authentication (SSO simulation)
def authenticate_user():
    # In a real app, implement SSO logic here (e.g., with OAuth)
    return True  # Simulate successful authentication

# Load the Iris dataset
@st.cache_resource
def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return pd.DataFrame(X, columns=feature_names), pd.Series(y), target_names

# Train a logistic regression model
@st.cache_resource
def train_model(X, y):
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)
    return model

# Detect model drift
def detect_drift(reference_data, current_data):
    # Define the column mapping
    column_mapping = ColumnMapping()

    # Create a report object
    report = Report(metrics=[DataDriftPreset()])

    # Generate the report
    report.run(current_data=current_data, reference_data=reference_data, column_mapping=column_mapping)

    # Save and display the report using in-memory tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp_file:
        report.save(tmp_file.name)
        tmp_file_path = tmp_file.name
        
        # Read the HTML content
        #with open(tmp_file.name, 'r') as f:
            #drift_report_html = f.read()

    # Display the report using an iframe
    st.components.v1.iframe(src=tmp_file_path, height=600, width=1000)

# Exploratory Data Analysis (EDA)
def perform_eda(df):
    st.subheader("Exploratory Data Analysis (EDA)")

    # EDA 1: Basic statistics
    st.write("**Basic Statistics**")
    st.write(df.describe())

    # EDA 2: Class distribution
    st.write("**Class Distribution**")
    class_dist = pd.DataFrame(df['species'].value_counts()).reset_index()
    class_dist.columns = ['Species', 'Count']
    st.bar_chart(class_dist.set_index('Species'))

    # EDA 3: Pairplot
    # Create the pairplot
    pairplot_fig = sns.pairplot(df, hue='species')
    
    # Render the plot using st.pyplot
    st.pyplot(pairplot_fig)

    # EDA 4: Correlation Heatmap
    st.write("**Correlation Heatmap**")
    # Exclude the non-numeric 'species' column for correlation matrix calculation
    numeric_df = df.drop(columns=['species'])
    
    # Create a new figure for the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    
    # Display the plot
    st.pyplot(fig)
    
    # Clear the figure to prevent overlap
    plt.clf()

    # EDA 5: Boxplot for each feature
    st.write("**Boxplots for Each Feature**")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    sns.boxplot(data=df, x='species', y='sepal length (cm)', ax=ax[0, 0])
    sns.boxplot(data=df, x='species', y='sepal width (cm)', ax=ax[0, 1])
    sns.boxplot(data=df, x='species', y='petal length (cm)', ax=ax[1, 0])
    sns.boxplot(data=df, x='species', y='petal width (cm)', ax=ax[1, 1])
    st.pyplot(fig)

# Streamlit app structure
def main():
    st.title("Iris Classification App")

    # Authenticate user
    if authenticate_user():
        st.success("User authenticated successfully!")

        # Load dataset
        df, y, target_names = load_iris_data()

        # Add species name to the dataframe for EDA purposes
        df['species'] = y.map({i: target_names[i] for i in range(len(target_names))})

        # Perform EDA
        perform_eda(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['species']), y, test_size=0.2, random_state=42)

        # Train model
        model = train_model(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)
        st.subheader("Model Performance")
        st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, predictions, target_names=target_names))

        # Simulate loading current dataset for drift detection
        current_data = df.drop(columns=['species']).copy()  # Here, we just reuse the original data for demonstration
        detect_drift(df.drop(columns=['species']), current_data)

if __name__ == "__main__":
    main()
