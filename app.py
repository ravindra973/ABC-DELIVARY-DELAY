import streamlit as st
import joblib
import pandas as pd

st.title('Delivery Delay Prediction')

# Load the trained model
model = joblib.load('Delivery_Delay.sav')

st.write("Enter the features below to predict delivery delay.")

# Create input widgets for each feature
delivery_distance = st.slider('Delivery Distance', min_value=0.0, max_value=100.0, value=50.0)
traffic_congestion = st.slider('Traffic Congestion (1-5)', min_value=1, max_value=5, value=3)
weather_condition = st.slider('Weather Condition (1-5)', min_value=1, max_value=5, value=3)
delivery_slot = st.slider('Delivery Slot (1-3)', min_value=1, max_value=3, value=2)
driver_experience = st.slider('Driver Experience (Years)', min_value=0, max_value=30, value=10)
num_stops = st.slider('Number of Stops', min_value=1, max_value=10, value=5)
vehicle_age = st.slider('Vehicle Age (Years)', min_value=0, max_value=15, value=5)
road_condition_score = st.slider('Road Condition Score (1-5)', min_value=1, max_value=5, value=3)
package_weight = st.slider('Package Weight (kg)', min_value=0.0, max_value=50.0, value=25.0)
fuel_efficiency = st.slider('Fuel Efficiency (km/l)', min_value=0.0, max_value=30.0, value=15.0)
warehouse_processing_time = st.slider('Warehouse Processing Time (minutes)', min_value=0, max_value=120, value=60)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([{
    'Delivery_Distance': delivery_distance,
    'Traffic_Congestion': traffic_congestion,
    'Weather_Condition': weather_condition,
    'Delivery_Slot': delivery_slot,
    'Driver_Experience': driver_experience,
    'Num_Stops': num_stops,
    'Vehicle_Age': vehicle_age,
    'Road_Condition_Score': road_condition_score,
    'Package_Weight': package_weight,
    'Fuel_Efficiency': fuel_efficiency,
    'Warehouse_Processing_Time': warehouse_processing_time
}])

if st.button('Predict Delivery Delay'):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    st.write(f"### Prediction: {'Delayed' if prediction[0] == 1 else 'On Time'}")
    st.write(f"### Probability of Delay: {prediction_proba[0]:.2f}")

st.markdown("---")
st.write("Note: This is a demo application. The model's performance may vary.")
