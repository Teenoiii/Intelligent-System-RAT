import streamlit as st

st.title("Data Preprocessing")
st.write("""
ในหน้านี้จะอธิบายกระบวนการเตรียมข้อมูลก่อนนำไปใช้กับโมเดล ML และ Neural Network
""")

st.subheader("1. ตรวจสอบข้อมูลที่ขาดหาย")
st.write("ใช้ `pandas` เพื่อตรวจสอบค่า null และเติมค่าที่ขาด")

st.subheader("2. การทำ Feature Scaling")
st.write("ใช้ `StandardScaler` หรือ `MinMaxScaler` ในการปรับขนาดข้อมูลให้อยู่ในช่วงที่เหมาะสม")
