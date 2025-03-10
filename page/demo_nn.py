import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib

# 📌 โหลดโมเดล
model = tf.keras.models.load_model("models/car_price_model.keras")

# 📌 โหลด Column Transformer
ct = joblib.load("models/column_transformer.pkl")

# 📌 ค่าเงินโดยประมาณ (อัปเดตอัตราจริงก่อนใช้งาน)
EXCHANGE_RATE = 0.42  # 1 รูปี ≈ 0.42 บาท

# ✅ ดึงรายชื่อยี่ห้อและรุ่นรถจาก Column Transformer
car_brands = list(ct.transformers_[1][1].categories_[2])  # ยี่ห้อรถ
car_models = list(ct.transformers_[1][1].categories_[3])  # รุ่นรถ

st.title("🚗 พยากรณ์ราคารถยนต์ด้วย AI")

# ✅ UI สำหรับป้อนข้อมูล
fuel_type = st.selectbox("ชนิดของเชื้อเพลิง", ["Petrol", "Diesel", "CNG", "Electric", "Lpg"])
transmission = st.selectbox("ระบบเกียร์", ["Manual", "Automatic"])
car_company = st.selectbox("ยี่ห้อรถยนต์", car_brands)

# ✅ แสดงเฉพาะรุ่นรถของยี่ห้อที่เลือก
filtered_models = [m for m in car_models if car_company.lower() in m.lower()]
if not filtered_models:  
    filtered_models = car_models  # ถ้าไม่มี ให้แสดงทุกรุ่น

car_model = st.selectbox("รุ่นรถยนต์ (Car Model)", filtered_models)

manufacture = st.number_input("ปีที่ผลิต", min_value=1990, max_value=2023, value=2020)
kms_driven = st.number_input("ระยะทางที่ขับไปแล้ว (กม.)", min_value=1000, max_value=500000, value=50000)
engine_cc = st.number_input("ขนาดเครื่องยนต์ (cc)", min_value=500, max_value=5000, value=1500)
seats = st.number_input("จำนวนที่นั่ง", min_value=2, max_value=10, value=5)
ownership = st.slider("ลำดับเจ้าของรถ", 1, 5, 1)

# ✅ ปุ่มพยากรณ์
if st.button("📊 ทำนายราคารถ"):
    try:
        # ✅ สร้าง DataFrame
        input_data = pd.DataFrame([[fuel_type, transmission, car_company, car_model, manufacture, kms_driven, engine_cc, seats, ownership]],
                                  columns=["fuel_type", "transmission", "car_company_name", "car_model_name",
                                           "manufacture", "kms_deriven_numeric", "engine_cc_numeric", "seats_numeric", "ownership_numeric"])

        # ✅ แปลงข้อมูลด้วย Column Transformer
        input_normalized = ct.transform(input_data)

        # ✅ ใช้โมเดลพยากรณ์
        predicted_price = model.predict(input_normalized)[0][0]

        # ✅ แปลง Lakh เป็น รูปี
        predicted_price_inr = predicted_price * 100000  

        # ✅ แปลงเป็น บาท
        predicted_price_thb = predicted_price_inr * EXCHANGE_RATE

        # ✅ แสดงผลลัพธ์
        st.success(f"🚗 ราคาที่คาดการณ์: {predicted_price_thb:,.2f} บาท")

    except Exception as e:
        st.error(f"🚨 เกิดข้อผิดพลาดในการพยากรณ์: {e}")
