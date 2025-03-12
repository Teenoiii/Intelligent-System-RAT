import streamlit as st
import os
import numpy as np
import joblib

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="ML & Neural Network Web App", layout="wide")

# Sidebar สำหรับเลือกหน้า
st.sidebar.title("Intelligent System Project")
st.sidebar.caption("Rat Kongkarat 6404062636447")
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊Data Preprocessing and algorithms", "📖 Model Explanation", "🤖 ML Demo", "🔬 NN Demo"])

# แสดงเนื้อหาตามหน้าที่เลือก
if page == "🏠 Home":
    exec(open("page/home_Introduction.py", encoding="utf-8").read())

elif page == "📊Data Preprocessing and algorithms":
    exec(open("page/data_preprocessing.py", encoding="utf-8").read())

elif page == "📖 Model Explanation":
    exec(open("page/model_explanation.py", encoding="utf-8").read())

elif page == "🤖 ML Demo":
    exec(open("page/demo_ml.py", encoding="utf-8").read())

elif page == "🔬 NN Demo":
    exec(open("page/demo_nn.py", encoding="utf-8").read())
