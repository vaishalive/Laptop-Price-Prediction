import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained pipeline
with open('model_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

# Load data used during training to get unique values for dropdowns
data = pd.read_csv("traineddata.csv")

st.title("💻 Laptop Price Predictor")

# Precompute the dictionary grouped by company
company_grouped_dict = {
    company: {
        "TypeName": group["TypeName"].unique().tolist(),
        "OpSys": group["OpSys"].unique().tolist(),
    }
    for company, group in data.groupby("Company")
}

# Company selectbox
selected_company = st.selectbox("Select a Company", list(company_grouped_dict.keys()))

# Show filtered options based on selected company
if selected_company:
    options = company_grouped_dict[selected_company]
    
    selected_type = st.selectbox("Laptop Type", options["TypeName"])
    selected_os = st.selectbox("OS", options["OpSys"])


# Ram present in laptop
ram = st.selectbox('Ram(GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight of laptop
weight = st.number_input('Weight of the laptop(KG)', min_value=0.5, max_value=5.0, step=0.1)

# touchscreen available in laptop or not
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size(Inches)', min_value=10.0, max_value=20.0, step=0.1)

# resolution of laptop
resolution = st.selectbox('Screen Resolution', [
                        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', data['CPU_name'].unique())

# hdd
hdd = st.selectbox('HDD(GB)', [0, 128, 256, 512, 1024, 2048])

# ssd
ssd = st.selectbox('SSD(GB)', [0, 8, 128, 256, 512, 1024])

#flash_storage
flash_storage = st.selectbox('Flash Storage(GB)', [0, 8, 128, 256, 512, 1024])

# gpu
gpu = st.selectbox('GPU(GB)', ['Intel', 'AMD', 'NVidia'])

if st.button('Predict Price'):
    # Validate that at least one storage option is selected
    if hdd == 0 and ssd == 0 and flash_storage == 0:
        st.error("❌ Please select at least one storage type: HDD, SSD, or Flash Storage.")
    else:
        try:
            # Convert to binary flags
            touchscreen_val = 1 if touchscreen == 'Yes' else 0
            ips_val = 1 if ips == 'Yes' else 0
            gpu = gpu.lower()  # Ensure GPU brand is lowercase
            type = selected_type.lower()  # Ensure TypeName is lowercase
            os = selected_os.lower()  # Ensure OpSys is lowercase
            cpu = cpu.lower()  # Ensure CPU name is lowercase
            company = selected_company.lower()  # Ensure Company is lowercase

            # Extract resolution and calculate PPI
            x_res = int(resolution.split('x')[0])
            y_res = int(resolution.split('x')[1])
            ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size

            # Create DataFrame with the expected columns
            input_df = pd.DataFrame([{
                'Ram': int(ram),
                'Weight': float(weight),
                'PPI': int(ppi),
                'HDD': int(hdd),
                'SSD': int(ssd),
                'Flash Storage': int(flash_storage),
                'Company': company,
                'TypeName': type,
                'OpSys': os,
                'CPU_name': cpu,
                'Gpu_Brand': gpu,
                'TouchScreen': int(touchscreen_val),
                'IPS': int(ips_val)
                }])

            # predict the price using the model
            predicted_price = model.predict(input_df)[0]
            predicted_price = round(float(predicted_price), 2)

            st.success(f"💰 Estimated Laptop Price: ₹{predicted_price - 1000} to ₹{predicted_price + 1000}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
