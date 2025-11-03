import streamlit as st
import pandas as pd
import requests
import joblib

# --- Configuration ---
# Get your FREE API key from https://openweathermap.org/
OPENWEATHER_API_KEY ="a4044e910fba01f1f9b30874fc7ca190"
# Coordinates for Kottayam (or your city)
LATITUDE = 9.5916
LONGITUDE = 76.5222

# --- Load Model and Scaler ---
try:
    model = joblib.load('api_model.pkl')
    scaler = joblib.load('api_scaler.pkl')
    model_columns = joblib.load('api_model_columns.pkl')
    print("API-compatible model loaded.")
except FileNotFoundError:
    st.error("Error: Model files not found! Please run `train_model_api.py` first.")
    st.stop()

# --- API Function ---
def get_live_weather_data(lat, lon, api_key):
    """Fetches and processes live weather data for the model."""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # --- Map API data to our model's features ---
        main = data.get('main', {})
        wind = data.get('wind', {})
        clouds = data.get('clouds', {})
        rain = data.get('rain', {})

        model_input = {
            'MinTemp': main.get('temp_min', 15.0),  # Default to 15 if not present
            'MaxTemp': main.get('temp_max', 25.0),  # Default to 25 if not present
            'Rainfall': rain.get('1h', 0.0),      # OWM gives '1h' rain. Default to 0
            'WindSpeed3pm': wind.get('speed', 5.0), # Use current 'speed' as proxy
            'Humidity3pm': main.get('humidity', 70.0), # Use current 'humidity'
            'Pressure3pm': main.get('pressure', 1010.0), # Use current 'pressure'
            'Cloud3pm': clouds.get('all', 50.0) / 12.5 # OWM is 0-100%, model wants 0-8. 
                                                     # 100 / 8 = 12.5. We scale it.
        }

        # Also return display data
        display_data = {
            "Location": data.get('name', 'Unknown'),
            "Temperature": f"{main.get('temp', 'N/A')}°C",
            "Description": data.get('weather', [{}])[0].get('description', 'N/A').capitalize(),
            "Humidity": f"{main.get('humidity', 'N/A')}%"
        }

        return model_input, display_data, None # No error

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            error_msg = "Error 401: Invalid API Key. Please check your OpenWeatherMap API key."
        else:
            error_msg = f"HTTP Error: {e}"
        return None, None, error_msg
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        return None, None, error_msg

# --- Streamlit UI ---
st.set_page_config(page_title="Rain Prediction App")
st.title("🌧️ Live Rain Prediction")
st.write("This app uses live OpenWeatherMap data to feed a model trained on matching features.")

st.markdown("---")

# --- Get and Display Live Data ---
model_input, display_data, error = get_live_weather_data(LATITUDE, LONGITUDE, OPENWEATHER_API_KEY)

if error:
    st.error(error)
    st.warning("Please make sure your `OPENWEATHER_API_KEY` is set correctly in the script.")
else:
    st.header(f"Live Weather in {display_data['Location']}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature", display_data["Temperature"])
    col2.metric("Description", display_data["Description"])
    col3.metric("Humidity", display_data["Humidity"])

    st.subheader("Model Prediction")

    # --- The ONLY manual input ---
    rain_today_input = st.selectbox(
        "Did it rain today?",
        ("No", "Yes"),
        help="This is the one feature the API can't provide and is crucial for the model."
    )

    # Add this manual input to our model_input dictionary
    model_input['RainToday'] = 1 if rain_today_input == "Yes" else 0

    if st.button("Predict Rain Tomorrow", type="primary"):
        try:
            # 1. Create a DataFrame in the correct order
            input_df = pd.DataFrame([model_input])
            input_df = input_df[model_columns] # Ensures correct column order

            # 2. Scale the input
            scaled_input = scaler.transform(input_df)

            # 3. Make prediction
            prediction = model.predict(scaled_input)
            probability = model.predict_proba(scaled_input)[0][1] # Get probability of 'Yes'

            # 4. Display result
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"**Prediction: It WILL RAIN tomorrow** (Probability: {probability*100:.2f}%)")
            else:
                st.success(f"**Prediction: It will NOT rain tomorrow** (Probability: {100 - (probability*100):.2f}%)")

            # Show the data fed into the model
            with st.expander("Show data fed to model"):
                st.write("This is the raw data (after mapping) that was sent to the model:")
                st.dataframe(input_df)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")