import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.title("📊Data Preprocessing and algorithm")

tabs = st.tabs(["Data Preprocessing", "Detail of DataSet()", "Detail of DataSet()", "Development Model", "Theory of ML Model", "Theory of NN Model"]) 

with tabs[0]:
    st.title("Data Preprocessing")
with tabs[1]:
    st.title("Detail of DataSet(ML)")
    st.write("ระบุที่มำของ Dataset (เช่น download จำกเว็บไซต์ใดก็ได้, สร้ำงโดย ChatGPT, หรือใช้ข้อมูลจริง) อธิบำย feature ของ Dataset")
with tabs[2]:
    st.title("Detail of DataSet(NN)")
with tabs[3]:
    st.title("Development Model")
with tabs[4]:
    st.title("Theory of ML Model")
with tabs[5]:  
    st.title("Theory of NN Model")


