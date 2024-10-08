import streamlit as st
from classification import load_iris_data, perform_eda, train_model, model_performance
from monitoring import Monitoring
from model import Model
from view import View
from sklearn.model_selection import train_test_split

def main():
    st.title("Iris Classification and Monitoring App")

    view = View()
    model = Model()
    monitoring = Monitoring(view, model)

    # Add option to select between classification or monitoring
    option = st.sidebar.selectbox('Select Application', ('Classification', 'Monitoring'))

    if option == 'Classification':
        st.write("Classification App")
        df, y, target_names = load_iris_data()
        df['species'] = y.map({i: target_names[i] for i in range(len(target_names))})

        perform_eda(df)

        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['species']), y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)

        model_performance(model, X_test, y_test, target_names)

    elif option == 'Monitoring':
        monitoring.run_monitoring()

if __name__ == "__main__":
    main()
