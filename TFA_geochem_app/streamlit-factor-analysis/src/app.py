import streamlit as st
import pandas as pd
from components.factor_analysis import perform_factor_analysis
from components.decision_tree import DecisionTreePAI
from utils.data_loader import load_data
from utils.preprocessing import preprocess_data

def main():
    st.title("Factor Analysis and Decision Tree PAI Application")

    # Load data
    data_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if data_file is not None:
        df = load_data(data_file)
        st.write("Data Loaded Successfully!")
        st.dataframe(df)

        # Preprocess data
        processed_data = preprocess_data(df)
        st.write("Data Preprocessed Successfully!")

        # Factor Analysis
        if st.button("Perform Factor Analysis"):
            factor_results = perform_factor_analysis(processed_data)
            st.write("Factor Analysis Results:")
            st.dataframe(factor_results)

        # Decision Tree PAI
        dt_pai = DecisionTreePAI()
        num_classes = st.number_input("Enter number of classes for PAI calculation:", min_value=1, value=2)
        dt_pai.setup_classes(num_classes)

        if st.button("Fit Decision Tree"):
            results = dt_pai.fit(processed_data, 'Label')
            st.write("Gini Index Results:")
            for attr, (gini_df, _, _) in results['gini_results'].items():
                st.write(f"{attr}:")
                st.dataframe(gini_df)

            st.write("PAI Results:")
            st.dataframe(results['pai_results'])

if __name__ == "__main__":
    main()