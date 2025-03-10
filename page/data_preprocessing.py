import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

st.title("📊Data Preprocessing and algorithm")

tabs = st.tabs(["Data Preprocessing", "Detail of DataSet()", "Detail of DataSet()", "Development Model", "Theory of ML Model", "Theory of NN Model"]) 

with tabs[0]:
    st.title("🔍Data Preprocessing (การเตรียมข้อมูลก่อนเทรนโมเดล)")
    st.subheader("ก่อนสร้างโมเดล เราต้องเตรียมข้อมูลให้เหมาะสม โดยทำตามขั้นตอนต่อไปนี้:")
    st.title("Machine Learning")
    st.subheader("ตรวจสอบ Missing Values และค่าผิดปกติ")
    st.write("-ถ้าข้อมูลขาดหาย (Missing Data) เช่น พื้นที่บ้านไม่ถูกระบุ → ใช้ค่าเฉลี่ย (mean) หรือค่ากลาง (median) เติมเต็ม")
    st.write("-ถ้าพบค่าผิดปกติ เช่น ราคาบ้าน = 0 → ลบออกหรือแก้ไข")
    st.subheader("Encoding ข้อมูลประเภทข้อความ (Categorical Data)")
    st.write("XGBoost ไม่รองรับข้อมูลข้อความโดยตรง เช่น Furnished,Unfurnished ดังนั้นต้องแปลงเป็นตัวเลข เช่น:")
    st.write("Furnished → 1")
    st.write("Unfurnished → 0")
    st.write("หรือใช้ One-Hot Encoding เพื่อสร้างคอลัมน์ใหม่แทนข้อความ")
    st.subheader("Scaling ข้อมูลเชิงตัวเลข")
    st.write("โมเดล XGBoost ไม่ต้องการการ Scaling แบบ Standardization หรือ Normalization เหมือน Neural Network แต่การปรับค่าให้สมดุลสามารถช่วยให้โมเดลเรียนรู้ได้เร็วขึ้น")
    st.subheader("Train-Test Split")
    st.write("แบ่งข้อมูลเป็น 2 ส่วน:")
    st.write("-Train Set (80%) → ใช้สำหรับฝึกโมเดล")
    st.write("-Test Set (20%) → ใช้สำหรับทดสอบโมเดล")


with tabs[1]:
    st.title("Detail of DataSet(ML)")
    st.header("ที่มาของ Dataset")
    st.write("Housing Prices Dataset เป็นชุดข้อมูลที่ใช้สำหรับวิเคราะห์และพยากรณ์ราคาบ้าน โดยรวบรวมข้อมูลเกี่ยวกับลักษณะของที่พักอาศัยและปัจจัยที่มีผลต่อราคา")
    st.write("อาทิเช่น ขนาดพื้นที่ จำนวนห้องนอน จำนวนห้องน้ำ และอื่น ๆ")
    st.markdown("แหล่งที่มา: [Kaggle - Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)")
    st.subheader("จุดประสงค์ของ Dataset")
    st.write("-ใช้สำหรับการสร้างแบบจำลอง Machine Learning เพื่อพยากรณ์ราคาที่พักอาศัย")
    st.write("-ศึกษาความสัมพันธ์ระหว่างปัจจัยต่าง ๆ กับราคาบ้าน")
    st.header("Detail of DataSet (ML)")
    st.subheader("โครงสร้างของ Dataset")
    st.write("ชุดข้อมูลนี้ประกอบด้วย 506 แถว และ 14 คอลัมน์ ซึ่งแสดงรายละเอียดของแต่ละคุณลักษณะได้ดังนี้:")

    df = pd.read_csv("Housing.csv")
    st.write("### ตัวอย่างข้อมูล")
    st.dataframe(df.head())

    st.write("### คำอธิบายแต่ละ Feature")
    table_html = """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }
        th {
            background-color: #222;
            color: white;
        }
        td {
            background-color: #333;
            color: white;
        }
    </style>

    <table>
        <tr>
            <th>Feature</th>
            <th>Description</th>
            <th>Data Type</th>
        </tr>
        <tr><td>Price</td><td>ราคาบ้าน (เป้าหมายที่ต้องพยากรณ์)</td><td>float</td></tr>
        <tr><td>Area</td><td>ขนาดพื้นที่ของบ้าน (ตร.ม.)</td><td>float</td></tr>
        <tr><td>Bedrooms</td><td>จำนวนห้องนอน</td><td>int</td></tr>
        <tr><td>Bathrooms</td><td>จำนวนห้องน้ำ</td><td>int</td></tr>
        <tr><td>Stories</td><td>จำนวนชั้นของบ้าน</td><td>int</td></tr>
        <tr><td>Mainroad</td><td>บ้านติดถนนใหญ่หรือไม่ (yes/no)</td><td>object</td></tr>
        <tr><td>Guestroom</td><td>มีห้องรับรองแขกหรือไม่ (yes/no)</td><td>object</td></tr>
        <tr><td>Basement</td><td>มีชั้นใต้ดินหรือไม่ (yes/no)</td><td>object</td></tr>
        <tr><td>Hotwaterheating</td><td>มีระบบน้ำร้อนหรือไม่ (yes/no)</td><td>object</td></tr>
        <tr><td>Airconditioning</td><td>มีเครื่องปรับอากาศหรือไม่ (yes/no)</td><td>object</td></tr>
        <tr><td>Parking</td><td>จำนวนที่จอดรถ</td><td>int</td></tr>
        <tr><td>Prefarea</td><td>อยู่ในพื้นที่ทำเลดีหรือไม่ (yes/no)</td><td>object</td></tr>
        <tr><td>Furnishingstatus</td><td>สถานะของเฟอร์นิเจอร์ (furnished/semi-furnished/unfurnished)</td><td>object</td></tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    


with tabs[2]:
    st.title("Detail of DataSet(NN)")
    st


with tabs[3]:
    st.title("🏗 Development Model (กระบวนการสร้างโมเดล)")
    st.title("Machine Learning")
    st.subheader("เลือกอัลกอริธึมของ XGBoost")
    st.write("XGBoost รองรับ 3 Algorithm:")
    st.write("-gbtree → Gradient Boosting Decision Trees (ค่าเริ่มต้น)")
    st.write("-gblinear → Linear Regression หรือ Logistic Regression")
    st.write("-dart → Dropout Trees (ลด Overfittin")
    st.subheader("ตั้งค่าพารามิเตอร์ของโมเดล")
    st.write("from xgboost import XGBRegressor")
    st.write(" booster=gbtree,       # ใช้ต้นไม้ตัดสินใจ (Decision Trees)")
    st.write(" n_estimators=500,       # จำนวนต้นไม้ที่ใช้")
    st.write(" learning_rate=0.05,     # อัตราการเรียนรู้")
    st.write(" max_depth=5,            # ความลึกของต้นไม้")
    st.write(" subsample=0.8,          # ใช้ข้อมูลสุ่ม 80% ในแต่ละรอบ")
    st.write(" colsample_bytree=0.6    # ใช้ฟีเจอร์สุ่ม 60% ในแต่ละต้นไม้")
    st.write(")")
    st.write("🔹 ค่าพารามิเตอร์เหล่านี้ถูกปรับโดย Hyperparameter Tuning เพื่อให้โมเดลทำงานได้ดีที่สุด")
    st.subheader("ฝึกโมเดล (Training Model)")
    st.write("xgb_model.fit(X_train, y_train")
    st.subheader("ประเมินผลโมเดล")
    st.write("from sklearn.metrics import mean_absolute_error")
    st.write("y_pred = xgb_model.predict(X_test)")
    st.write("mae = mean_absolute_error(y_test, y_pred)")
    st.write("print(f 📊 Mean Absolute Error (MAE): {mae:.2f} )")
with tabs[4]:
    st.title("🧠Theory of ML Model")
    st.subheader("Gradient Boosting Decision Trees (GBDT)")
    st.markdown("XGBoost เป็นโมเดลที่ใช้เทคนิค Gradient Boosting Decision Trees (GBDT) ซึ่งทำงานโดย:")
    st.write("1.สร้าง ต้นไม้ตัดสินใจ (Decision Trees)")
    st.write("2.ปรับปรุงข้อผิดพลาดของแต่ละต้นไม้ โดยใช้ Gradient Descent")
    st.write("3.รวมต้นไม้หลายต้นเพื่อให้ได้ผลลัพธ์ที่แม่นยำขึ้น")
    st.header("ทำไมถึงใช้ XGBoost ไม่ใช้ตัวอื่น เช่น Random Forest")
    st.subheader("ความแตกต่างระหว่าง XGBoost กับ Random Forest")
    # ใช้ HTML + CSS ใน markdown
    table_html = """
    <style>
        table {
        width: 100%;
        border-collapse: collapse;
        }
        th, td {
        border: 1px solid #ddd;
        padding: 10px;
        text-align: left;
        }
        th {
        background-color: #222;
        color: white;
        }
        td {
        background-color: #333;
        color: white;
        }
    </style>

    <table>
        <tr>
            <th>Feature</th>
            <th>XGBoost</th>
            <th>Random Forest</th>
        </tr>
        <tr>
            <td>การเรียนรู้</td>
            <td>เรียนรู้ลึกด้วยแบบ Boosting</td>
            <td>ฝึกต้นไม้พร้อมกันแบบ Bagging</td>
        </tr>
        <tr>
            <td>ความเร็ว</td>
            <td>เร็วกว่าเพราะใช้ Optimized Gradient Boosting</td>
            <td>ช้ากว่า</td>
        </tr>
        <tr>
            <td>การจัดการ Missing Values</td>
            <td>รองรับได้อัตโนมัติ</td>
            <td>ต้องเติมค่า Missing Data</td>
        </tr>
        <tr>
            <td>การทำงานกับข้อมูลขนาดใหญ่</td>
            <td>เหมาะสมมาก</td>
            <td>ลาช้ากว่า</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.write("")
    st.write("")

with tabs[5]:  
    st.title("Theory of NN Model")


