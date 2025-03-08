import streamlit as st
import pickle
import pandas as pd

st.title("Machine Learning Model Demo")

# โหลดโมเดลที่ฝึกไว้แล้ว
with open("models/ml_model.pkl", "rb") as f:
    model = pickle.load(f)

st.subheader("ทดสอบโมเดล Machine Learning")
input_data = st.text_input("กรอกค่าทดสอบ (เช่น 5.1,3.5,1.4,0.2)")

if st.button("Predict"):
    try:
        input_list = list(map(float, input_data.split(",")))
        df = pd.DataFrame([input_list])
        prediction = model.predict(df)
        st.write(f"ผลลัพธ์ที่คาดการณ์: {prediction[0]}")
    except:
        st.write("กรุณาใส่ค่าข้อมูลให้ถูกต้อง")
