import streamlit as st
import pandas as pd
import time
from sklearn import datasets

class Monitoring:
    def __init__(self, view, model):
        self.view = view
        self.model = model

    def run_monitoring(self):
        st.title("Data & Model Monitoring App")
        st.write("This app will help you monitor model performance, data drift, and data quality.")

        # Select which reports to generate
        st.subheader("Select Reports to Generate")
        generate_model_report = st.checkbox("Generate Model Performance Report")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_data_quality = st.checkbox("Generate Data Quality Report")

        if st.button("Submit"):
            st.write("Loading the Iris dataset...")

            # Load Iris dataset from sklearn
            data_start = time.time()
            iris = datasets.load_iris()
            iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
            iris_df['target'] = iris.target
            data_end = time.time()
            time_taken = data_end - data_start

            st.write(f"Loaded the Iris dataset in {time_taken:.2f} seconds")

            # Split the data into reference and current batches (50/50 split)
            reference_data = iris_df.sample(frac=0.5, random_state=42)
            current_data = iris_df.drop(reference_data.index)

            # Display and train model on the split data
            self.view.display_monitoring(reference_data, current_data)
            self.model.train_model(reference_data, current_data)

            # Generate selected reports
            if generate_model_report:
                st.write("### Model Performance Report")
                performance_report = self.model.performance_report(reference_data, current_data)
                self.view.display_performance_report(performance_report, "Model Performance Report")

            if generate_target_drift:
                st.write("### Target Drift Report")
                target_report = self.model.target_report(reference_data, current_data)
                self.view.display_report(target_report, "Target Drift Report")

            if generate_data_drift:
                st.write("### Data Drift Report")
                data_drift_report = self.model.data_drift_report(reference_data, current_data)
                self.view.display_report(data_drift_report, "Data Drift Report")

            if generate_data_quality:
                st.write("### Data Quality Report")
                data_quality_report = self.model.data_quality_report(reference_data, current_data)
                self.view.display_report(data_quality_report, "Data Quality Report")
