import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time

# โหลดโมเดล XGBoost
model_path = "models/xgboost_best_model.pkl"
model = joblib.load(model_path)

# โหลด Dataset เพื่อตรวจสอบค่าต่ำสุด-สูงสุดของขนาดพื้นที่
df = pd.read_csv("Housing.csv")
area_min = int(df["area"].min())
area_max = int(df["area"].max())

st.title("🏠 Machine Learning Model Demo - Predict California House Price") 

st.write("📌 **กรอกค่าคุณสมบัติของบ้านเพื่อทำนายราคา**")

# ✅ **เพิ่มคำอธิบายฟีเจอร์ + ปรับ UI**
area = st.number_input(f"🏠 ขนาดพื้นที่ (ตร.ม.) (ระหว่าง {area_min} - {area_max})", 
                        min_value=area_min, max_value=area_max, value=(area_min + area_max) // 2)

bedrooms = st.slider("🛏 จำนวนห้องนอน", 1, 10, 3)
bathrooms = st.slider("🚿 จำนวนห้องน้ำ", 1, 10, 2)
stories = st.slider("🏢 จำนวนชั้นของบ้าน", 1, 5, 2)
parking = st.slider("🚗 จำนวนที่จอดรถ", 0, 5, 1)

# ✅ **ตัวเลือกแบบ Yes/No (ใช้ 1/0)**
mainroad_yes = st.checkbox("🛣 บ้านติดถนนใหญ่", value=True)
guestroom_yes = st.checkbox("🛋 มีห้องรับแขก", value=False)
basement_yes = st.checkbox("🔽 มีห้องใต้ดิน", value=False)
hotwaterheating_yes = st.checkbox("🔥 มีเครื่องทำน้ำร้อน", value=False)
airconditioning_yes = st.checkbox("❄ มีแอร์", value=True)
prefarea_yes = st.checkbox("🌳 อยู่ในทำเลดี", value=True)

# ✅ **สถานะเฟอร์นิเจอร์ (ใช้ Dropdown)**
furnishing_status = st.selectbox("🛋 สถานะเฟอร์นิเจอร์", ["ไม่มีเฟอร์นิเจอร์", "มีบางส่วน", "มีครบ"])
furnishingstatus_semi = 1 if furnishing_status == "มีบางส่วน" else 0
furnishingstatus_unfurnished = 1 if furnishing_status == "ไม่มีเฟอร์นิเจอร์" else 0

# ✅ **คำนวณฟีเจอร์ใหม่**
bathroom_ratio = round(bathrooms / bedrooms, 2) if bedrooms > 0 else 0
area_per_story = round(area / stories, 2) if stories > 0 else 0

# ✅ **ปุ่มพยากรณ์**
if st.button("📊 Predict Price"):
    try:
        # ✅ **สร้าง placeholder สำหรับแสดงผล**
        placeholder = st.empty()

        placeholder.markdown("⌛ **กำลังคำนวณราคา...**")
        time.sleep(1)

        # ✅ **แสดง Progress Bar**
        placeholder.progress(0, "เตรียมข้อมูล...")
        time.sleep(1)
        placeholder.progress(50, "กำลังประมวลผลโมเดล...")
        time.sleep(1)
        placeholder.progress(100, "กำลังแสดงผลลัพธ์...")
        time.sleep(1)

        # ✅ **สร้าง DataFrame ให้ตรงกับฟีเจอร์ที่โมเดลต้องการ**
        input_data = np.array([[area, bedrooms, bathrooms, stories, parking,
                                int(mainroad_yes), int(guestroom_yes), int(basement_yes),
                                int(hotwaterheating_yes), int(airconditioning_yes), int(prefarea_yes),
                                furnishingstatus_semi, furnishingstatus_unfurnished, bathroom_ratio, area_per_story]])

        df_input = pd.DataFrame(input_data, columns=model.feature_names_in_)

        # ✅ **ใช้โมเดลพยากรณ์**
        prediction = model.predict(df_input)[0]

        # ✅ **แสดงผลลัพธ์**
        placeholder.success(f"🏠 ราคาบ้านที่คาดการณ์: {prediction:,.2f} บาท")
        time.sleep(2)

        # ✅ **เคลียร์ข้อความ**
        placeholder.empty()

    except Exception as e:
        st.error(f"⚠️ เกิดข้อผิดพลาด: {e}")
