import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import io
import openpyxl

@st.cache_data(max_entries=10)
def load_and_prepare_data(data):
    features = ['lat', 'lon', 'elevation', 'temperature', 'humidity', 'pressure']
    X = data[features]
    y = data['rainfall']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

@st.cache_resource(max_entries=3)
def create_and_train_ann_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=32, callbacks=[early_stopping], verbose=0)
    return model, history

@st.cache_resource(max_entries=3)
def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    return model

@st.cache_resource(max_entries=3)
def train_random_forest_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return f"{model_name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}"

def create_visualizations(data, y_true, y_pred):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    ax1.scatter(y_true, y_pred)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Rainfall')
    ax1.set_ylabel('Predicted Rainfall')
    ax1.set_title('Actual vs Predicted Rainfall')
    
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=30)
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Residuals')
    
    corr = data[['lat', 'lon', 'elevation', 'temperature', 'humidity', 'pressure', 'rainfall']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)
    ax3.set_title('Feature Correlation Heatmap')
    
    return fig

def main():
    st.set_page_config(page_title="Rainfall Estimation Dashboard", layout="wide")
    
    st.title("Rainfall Estimation Dashboard")
    
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
            
            X, y, scaler = load_and_prepare_data(data)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            
            with st.spinner("Training models... This may take a few minutes."):
                ann_model, history = create_and_train_ann_model(X_train, y_train, X_val, y_val)
                xgb_model = train_xgboost_model(X_train, y_train)
                rf_model = train_random_forest_model(X_train, y_train)
            
            y_pred_ann = ann_model.predict(X_test).flatten()
            y_pred_xgb = xgb_model.predict(X_test)
            y_pred_rf = rf_model.predict(X_test)
            y_pred_ensemble = (y_pred_ann + y_pred_xgb + y_pred_rf) / 3
            
            residuals = y_test - y_pred_ensemble
            residual_model = train_xgboost_model(X_test, residuals)
            y_pred_final = y_pred_ensemble + residual_model.predict(X_test)
            
            st.subheader("Model Evaluation")
            col1, col2 = st.columns(2)
            with col1:
                st.write(evaluate_model(y_test, y_pred_ann, "ANN"))
                st.write(evaluate_model(y_test, y_pred_xgb, "XGBoost"))
            with col2:
                st.write(evaluate_model(y_test, y_pred_rf, "Random Forest"))
                st.write(evaluate_model(y_test, y_pred_ensemble, "Ensemble"))
            st.write(evaluate_model(y_test, y_pred_final, "Final Model (with residual correction)"))
            
            st.subheader("Visualizations")
            fig = create_visualizations(data, y_test, y_pred_final)
            st.pyplot(fig)
            
            st.subheader("Predict Missing Rainfall")
            col1, col2, col3 = st.columns(3)
            with col1:
                lat = st.number_input("Latitude", value=5.6)
                lon = st.number_input("Longitude", value=105.2)
            with col2:
                elevation = st.number_input("Elevation", value=50)
                temperature = st.number_input("Temperature", value=28)
            with col3:
                humidity = st.number_input("Humidity", value=80)
                pressure = st.number_input("Pressure", value=1010)
            
            if st.button("Predict"):
                missing_data = np.array([[lat, lon, elevation, temperature, humidity, pressure]])
                missing_data_scaled = scaler.transform(missing_data)
                ann_pred = ann_model.predict(missing_data_scaled).flatten()
                xgb_pred = xgb_model.predict(missing_data_scaled)
                rf_pred = rf_model.predict(missing_data_scaled)
                ensemble_pred = (ann_pred + xgb_pred + rf_pred) / 3
                residual_pred = residual_model.predict(missing_data_scaled)
                final_pred = ensemble_pred + residual_pred
                st.success(f"Predicted rainfall: {final_pred[0]:.2f} mm")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    else:
        st.info("Please upload a CSV or Excel file to begin.")

if __name__ == "__main__":
    main()
