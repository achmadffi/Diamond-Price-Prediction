import pandas as pd
import numpy as np
import pickle
import streamlit as st
import streamlit.components.v1 as stc
from datetime import datetime

# Load model & preprocessing
def load_resources():
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('ordinal_encoder.pkl', 'rb') as f:
        ordinal_encoder = pickle.load(f)
    with open('power_transformers.pkl', 'rb') as f:
        power_transformers = pickle.load(f)
    
    return {
        'xgb_model': xgb_model,
        'ordinal_encoder': ordinal_encoder,
        'power_transformers': power_transformers
    }

def predict_price(models, input_dict):
    categorical_cols = ['cut', 'color', 'clarity']
    numerical_cols = ['carat', 'table', 'x', 'y', 'z']
    
    df = pd.DataFrame([input_dict])
    
    encoder = models['ordinal_encoder']
    model = models['xgb_model']
    
    df_cat = encoder.transform(df[categorical_cols])
    df_cat = pd.DataFrame(df_cat, columns=categorical_cols)
    
    df_final = pd.concat([df_cat, df[numerical_cols].reset_index(drop=True)], axis=1)
    
    prediction = model.predict(df_final)
    return float(prediction[0])

def main():
    st.set_page_config(
        page_title="Diamond Price Prediction",
        page_icon="💎",
        layout="centered"
    )

    # Inisialisasi session state
    if 'resources' not in st.session_state:
        st.session_state.resources = load_resources()
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=[
            'Timestamp', 'Carat', 'Cut', 'Color', 'Clarity', 
            'Table', 'Length', 'Width', 'Depth', 'Predicted Price'
        ])

    # Navigasi sederhana
    page = st.sidebar.radio("Menu", ["Prediction", "History"])
    
    if page == "Prediction":
        render_prediction_page()
    elif page == "History":
        render_history_page()

def render_prediction_page():
    st.title("💎 Diamond Price Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            carat = st.number_input("Carat Weight", min_value=0.1, max_value=10.0, step=0.1)
            cut = st.selectbox("Cut Quality (fair terendah - ideal tertinggi)", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
            color = st.selectbox("Color Grade (J terendah - D tertinggi)", ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
            clarity = st.selectbox("Clarity Grade (I1 terendah - IF tertinggi)", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
        
        with col2:
            table = st.number_input("Table", min_value=50.0, max_value=80.0, step=0.1)
            x = st.number_input("Length", min_value=0.1, max_value=11.0, step=0.1)
            y = st.number_input("Width", min_value=0.1, max_value=11.0, step=0.1)
            z = st.number_input("Depth", min_value=0.1, max_value=11.0, step=0.1)
        
        submitted = st.form_submit_button("Predict Price", type="primary")
        
        if submitted:
            input_dict = {
                'carat': carat,
                'cut': cut,
                'color': color,
                'clarity': clarity,
                'table': table,
                'x': x,
                'y': y,
                'z': z
            }

            with st.spinner("Calculating price..."):
                predicted_price = predict_price(st.session_state.resources, input_dict)

            # Tambah ke history
            new_entry = {
                'Timestamp': datetime.now(),
                'Carat': carat,
                'Cut': cut,
                'Color': color,
                'Clarity': clarity,
                'Table': table,
                'Length': x,
                'Width': y,
                'Depth': z,
                'Predicted Price': predicted_price
            }
            st.session_state.history = pd.concat([
                st.session_state.history,
                pd.DataFrame([new_entry])
            ], ignore_index=True)

            st.success(f"💎 Predicted Price: ${predicted_price:,.2f}")

def render_history_page():
    st.title("📊 Prediction History")

    if st.session_state.history.empty:
        st.warning("No prediction history yet!")
    else:
        history_display = st.session_state.history.copy()
        history_display['Predicted Price'] = history_display['Predicted Price'].apply(lambda x: f"${x:,.2f}")

        st.dataframe(
            history_display.sort_values('Timestamp', ascending=False),
            hide_index=True,
            use_container_width=True
        )

        csv = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name='diamond_price_predictions.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    main()
