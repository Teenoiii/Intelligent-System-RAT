import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

st.title("Neural Network Model Demo")

# โหลดโมเดลที่ฝึกไว้แล้ว
model = keras.models.load_model("models/nn_model.h5")

st.subheader("ทดสอบโมเดล Neural Network")
input_data = st.text_input("กรอกค่าทดสอบ (เช่น 0.5,0.2,0.3,0.7)")

if st.button("Predict"):
    try:
        input_list = np.array(list(map(float, input_data.split(",")))).reshape(1, -1)
        prediction = model.predict(input_list)
        st.write(f"ผลลัพธ์ที่คาดการณ์: {prediction}")
    except:
        st.write("กรุณาใส่ค่าข้อมูลให้ถูกต้อง")
