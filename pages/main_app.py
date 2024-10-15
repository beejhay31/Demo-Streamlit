import streamlit as st
from my_app.classification import load_iris_data, perform_eda, train_model, model_performance
from my_app.monitoring import Monitoring
from my_app.model import Model
from my_app.view import View
from sklearn.model_selection import train_test_split
import pandas as pd

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

        # Allow user input for prediction
        st.write("Input features to make a prediction")
        sepal_length = st.number_input("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.1, step=0.1)
        sepal_width = st.number_input("Sepal Width (cm)", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
        petal_length = st.number_input("Petal Length (cm)", min_value=1.0, max_value=7.0, value=1.4, step=0.1)
        petal_width = st.number_input("Petal Width (cm)", min_value=0.1, max_value=3.0, value=0.2, step=0.1)

        # Collect the input values into a DataFrame for prediction
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], 
                                  columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

        # Prediction button
        if st.button("Predict"):
            prediction = model.predict(input_data)[0]
            predicted_class = target_names[prediction]
            st.write(f"The predicted class is: **{predicted_class}**")

    elif option == 'Monitoring':
        # Display model performance in the monitoring interface
        st.write("Model Performance Report")
        df, y, target_names = load_iris_data()
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
        model_performance(model, X_test, y_test, target_names)
        
        st.write("Monitoring Interface")
        monitoring.run_monitoring()



if __name__ == "__main__":
    main_app()
