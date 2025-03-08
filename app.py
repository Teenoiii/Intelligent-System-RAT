import streamlit as st
import os

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML & Neural Network Web App", layout="wide")

# Sidebar สำหรับเลือกหน้า
st.sidebar.title("Intelligent System Project")
st.sidebar.caption("Rat Kongkarat 6404062636447")
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Data Preprocessing", "📖 Model Explanation", "🤖 ML Demo", "🔬 NN Demo"])

# แสดงเนื้อหาตามหน้าที่เลือก
if page == "🏠 Home":
    st.title("Welcome to ML & Neural Network Web App")
    st.write("เว็บแอปนี้ช่วยให้คุณเข้าใจและทดลองโมเดล Machine Learning และ Neural Network")

elif page == "📊 Data Preprocessing":
    exec(open("page/1_data_preprocessing.py", encoding="utf-8").read())

elif page == "📖 Model Explanation":
    exec(open("page/2_model_explanation.py", encoding="utf-8").read())

elif page == "🤖 ML Demo":
    exec(open("page/3_demo_ml.py", encoding="utf-8").read())

elif page == "🔬 NN Demo":
    exec(open("page/4_demo_nn.py", encoding="utf-8").read())
