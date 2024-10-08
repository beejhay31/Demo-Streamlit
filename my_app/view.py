import streamlit as st

class View:
    @staticmethod
    def display_report(report, report_name):
        """Display a report in the Streamlit app."""
        st.subheader(report_name)
        
        # Check if the report is an instance of Evidently's Report class
        if hasattr(report, 'show'):
            report.show()  # Use the show method to display the report
        else:
            st.write(report) 
            
    @staticmethod
    def display_monitoring(reference_data, current_data):
        """Display monitoring data."""
        st.subheader("Reference Data")
        st.write(reference_data)

        st.subheader("Current Data")
        st.write(current_data)
