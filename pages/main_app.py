import streamlit as st
from my_app.classification import load_iris_data, perform_eda, train_model, model_performance
from my_app.monitoring import Monitoring
from my_app.model import Model
from my_app.view import View
from sklearn.model_selection import train_test_split

# Main app interface after login
def main_app():
    st.title("Iris Classification and Monitoring App")

    view = View()
    model = Model()
    monitoring = Monitoring(view, model)

    # Add option to select between classification or monitoring
    option = st.sidebar.selectbox('Select Application', ('Exploratory Data Analysis (EDA)', 'Classification', 'Monitoring'))

    if option == 'Exploratory Data Analysis (EDA)':
        st.write("EDA Interface")
        df, y, target_names = load_iris_data()
        df['species'] = y.map({i: target_names[i] for i in range(len(target_names))})

        perform_eda(df)

    elif option == 'Classification':
        st.write("Classification Interface")
        df, y, target_names = load_iris_data()
        df['species'] = y.map({i: target_names[i] for i in range(len(target_names))})

        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['species']), y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        model_performance(model, X_test, y_test, target_names)

    elif option == 'Monitoring':
        st.write("Monitoring Interface")
        monitoring.run_monitoring()
