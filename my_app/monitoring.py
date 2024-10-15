import streamlit as st
import pandas as pd
import time
from sklearn import datasets

class Monitoring:
    def __init__(self, view, model):
        self.view = view
        self.model = model
        self.history = []  # To store monitoring history

    def run_monitoring(self):
        st.title("Data & Model Monitoring App")
        st.write("This app will help you monitor model performance, data drift, and data quality.")

        st.subheader("Select Reports to Generate")
        generate_model_report = st.checkbox("Generate Model Performance Report")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_data_quality = st.checkbox("Generate Data Quality Report")

        if st.button("Submit"):
            st.write("Loading the Iris dataset...")

            data_start = time.time()
            iris = datasets.load_iris()
            iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
            iris_df['target'] = iris.target
            data_end = time.time()
            time_taken = data_end - data_start

            st.write(f"Loaded the Iris dataset in {time_taken:.2f} seconds")

            reference_data = iris_df.sample(frac=0.5, random_state=42)
            current_data = iris_df.drop(reference_data.index)

            # Display and train model
            self.view.display_monitoring(reference_data, current_data)
            self.model.train_model(reference_data, current_data)

            # Store history of reports
            history_entry = {
                "model_performance": None,
                "target_drift": None,
                "data_drift": None,
                "data_quality": None,
                "timestamp": time.ctime()
            }

            # Generate reports and add to history
            if generate_model_report:
                st.write("### Model Performance Report")
                performance_report = self.model.performance_report(reference_data, current_data)
                self.view.display_performance_report(performance_report, "Model Performance Report")
                history_entry["model_performance"] = performance_report

            if generate_target_drift:
                st.write("### Target Drift Report")
                target_report = self.model.target_report(reference_data, current_data)
                self.view.display_report(target_report, "Target Drift Report")
                history_entry["target_drift"] = target_report

            if generate_data_drift:
                st.write("### Data Drift Report")
                data_drift_report = self.model.data_drift_report(reference_data, current_data)
                self.view.display_report(data_drift_report, "Data Drift Report")
                history_entry["data_drift"] = data_drift_report

            if generate_data_quality:
                st.write("### Data Quality Report")
                data_quality_report = self.model.data_quality_report(reference_data, current_data)
                self.view.display_report(data_quality_report, "Data Quality Report")
                history_entry["data_quality"] = data_quality_report

            # Save to history
            self.history.append(history_entry)

    def show_history(self):
        st.title("Monitoring History")
        if not self.history:
            st.write("No history available.")
            return
        
        for entry in self.history:
            st.write(f"Report generated on {entry['timestamp']}")
            if entry["model_performance"]:
                st.write("### Model Performance Report")
                self.view.display_performance_report(entry["model_performance"], "Model Performance Report")
            if entry["target_drift"]:
                st.write("### Target Drift Report")
                self.view.display_report(entry["target_drift"], "Target Drift Report")
            if entry["data_drift"]:
                st.write("### Data Drift Report")
                self.view.display_report(entry["data_drift"], "Data Drift Report")
            if entry["data_quality"]:
                st.write("### Data Quality Report")
                self.view.display_report(entry["data_quality"], "Data Quality Report")
