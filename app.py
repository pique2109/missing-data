import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.set_page_config(page_title="Rainfall Data Analyzer", layout="wide")
    
    st.title("Rainfall Data Analyzer")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            st.subheader("Raw Data")
            st.write(data.head())
            
            st.subheader("Data Statistics")
            st.write(data.describe())
            
            st.subheader("Data Visualization")
            
            # Simple bar chart of average rainfall by location
            if 'location' in data.columns and 'rainfall' in data.columns:
                avg_rainfall = data.groupby('location')['rainfall'].mean().sort_values(ascending=False)
                st.bar_chart(avg_rainfall)
                st.write("Average Rainfall by Location")
            
            # Simple line chart of rainfall over time
            if 'date' in data.columns and 'rainfall' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                daily_rainfall = data.groupby('date')['rainfall'].sum()
                st.line_chart(daily_rainfall)
                st.write("Daily Rainfall Over Time")
            
            st.subheader("Simple Rainfall Prediction")
            st.write("Enter values to get a simple rainfall prediction:")
            
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.number_input("Temperature", value=25.0)
                humidity = st.number_input("Humidity", value=60.0)
            with col2:
                pressure = st.number_input("Pressure", value=1013.0)
                elevation = st.number_input("Elevation", value=100.0)
            
            if st.button("Predict"):
                # This is a very simplistic prediction model for demonstration purposes
                predicted_rainfall = (temperature * 0.1 + humidity * 0.2 + pressure * 0.01 + elevation * 0.05) / 10
                st.success(f"Predicted rainfall: {predicted_rainfall:.2f} mm")
                st.write("Note: This is a simplistic prediction and should not be used for actual forecasting.")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    else:
        st.info("Please upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    main()
