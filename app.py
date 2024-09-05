import streamlit as st
import numpy as np
import pickle
import os

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Create a Streamlit app
st.title("Crop Recommendation System")

# Inputs from user
st.header("Input the required details:")
N = st.number_input('Nitrogen Content in Soil', min_value=0.0, max_value=100.0, value=50.0)
P = st.number_input('Phosphorus Content in Soil', min_value=0.0, max_value=100.0, value=50.0)
K = st.number_input('Potassium Content in Soil', min_value=0.0, max_value=100.0, value=50.0)
temp = st.number_input('Temperature (in °C)', min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input('Humidity (in %)', min_value=0.0, max_value=100.0, value=50.0)
ph = st.number_input('pH level of Soil', min_value=0.0, max_value=14.0, value=7.0)
rainfall = st.number_input('Rainfall (in mm)', min_value=0.0, max_value=300.0, value=100.0)

# When the user clicks predict
if st.button("Predict the Best Crop"):
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scaling the inputs
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    
    # Predicting the crop
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right now.".format(crop)
        
        # Displaying the result
        st.success(result)
        
        # Define path to the crop's image in its respective subfolder
        strings = ["images", crop , "Image_1.jpg"]
        image_path = "/".join(strings)
       



       
        
        # Check if the image exists and display it
        if os.path.exists(image_path):
            st.markdown(
                f"""
                <div style="text-align: center;">
                 <img src="{image_path}" alt="{crop}" width="400">
                <p>{crop}</p>
                </div>
                """,
                    unsafe_allow_html=True
                    )


        else:
            st.warning(f"Image for {crop} not found.")
    else:
        result = "Sorry, we could not determine the best crop with the provided data."
        st.error(result)
