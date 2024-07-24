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
import base64
import openpyxl
from lxml import etree

@st.cache_data
def load_and_prepare_data(data):
    features = ['lat', 'lon', 'elevation', 'temperature', 'humidity', 'pressure']
    X = data[features]
    y = data['rainfall']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

@st.cache_resource
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

@st.cache_resource
def train_xgboost_model(X_train, y_train):
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
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

def create_template_data():
    """Membuat data template."""
    return pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=10),
        'lat': np.random.uniform(0, 10, 10),
        'lon': np.random.uniform(100, 110, 10),
        'elevation': np.random.uniform(0, 1000, 10),
        'temperature': np.random.uniform(20, 35, 10),
        'humidity': np.random.uniform(60, 90, 10),
        'pressure': np.random.uniform(1000, 1020, 10),
        'rainfall': np.random.uniform(0, 100, 10)
    })

def get_csv_download_link(df, filename="template_data.csv"):
    """Menghasilkan link unduhan untuk file CSV."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Template</a>'
    return href

def get_excel_download_link(df, filename="template_data.xlsx"):
    """Menghasilkan link unduhan untuk file Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel Template</a>'
    return href

def get_xml_download_link(df, filename="template_data.xml"):
    """Menghasilkan link unduhan untuk file XML."""
    root = etree.Element("data")
    for _, row in df.iterrows():
        record = etree.SubElement(root, "record")
        for column, value in row.items():
            field = etree.SubElement(record, column)
            field.text = str(value)
    
    xml_data = etree.tostring(root, pretty_print=True, encoding='unicode')
    b64 = base64.b64encode(xml_data.encode('utf-8')).decode()
    href = f'<a href="data:application/xml;base64,{b64}" download="{filename}">Download XML Template</a>'
    return href

def main():
    st.set_page_config(page_title="Rainfall Estimation Dashboard", layout="wide")
    
    st.title("Rainfall Estimation Dashboard")
    
    # Tambahkan pilihan untuk mengunduh template
    st.subheader("Download Template")
    template_df = create_template_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(get_csv_download_link(template_df), unsafe_allow_html=True)
    with col2:
        st.markdown(get_excel_download_link(template_df), unsafe_allow_html=True)
    with col3:
        st.markdown(get_xml_download_link(template_df), unsafe_allow_html=True)
    
    st.write("Click one of the links above to download a template with sample data.")
    
    uploaded_file = st.file_uploader("Choose a CSV, Excel, or XML file", type=['csv', 'xlsx', 'xml'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                data = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.xml'):
                tree = etree.parse(uploaded_file)
                root = tree.getroot()
                data = []
                for record in root.findall('record'):
                    row = {child.tag: child.text for child in record}
                    data.append(row)
                data = pd.DataFrame(data)
                data['date'] = pd.to_datetime(data['date'])
                for col in ['lat', 'lon', 'elevation', 'temperature', 'humidity', 'pressure', 'rainfall']:
                    data[col] = pd.to_numeric(data[col])
            
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
        st.info("Please upload a CSV, Excel, or XML file to begin.")

if __name__ == "__main__":
    main()
