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
    st.write("โมเดลที่พัฒนาเป็น Machine Learning Model: XGBoost ซึ่งเป็นอัลกอริทึมประเภท Gradient Boosting ที่มีประสิทธิภาพสูงสำหรับปัญหาการพยากรณ์ (Regression) โดยขั้นตอนการเตรียมข้อมูลมีดังนี้:")
    st.subheader("✅ 1.การจัดการค่าที่หายไป (Handling Missing Values)")
    st.write("- ตรวจสอบว่าข้อมูลมี Missing Values หรือไม่ และใช้ Mean Imputation หรือ Mode Imputation แทนค่าที่ขาดหายไป")
    st.subheader("✅ 2.การแปลงฟีเจอร์ประเภทข้อความ (Categorical Encoding)")
    st.write("- ฟีเจอร์บางตัวเป็น หมวดหมู่ (Categorical Data) เช่น Furnishing Status, Main Road, Guest Room ซึ่งโมเดล XGBoost ไม่สามารถใช้ได้โดยตรง จำเป็นต้องแปลงเป็นตัวเลขก่อน")
    st.write("- ใช้ One-Hot Encoding กับฟีเจอร์ที่มีค่าหลายประเภทเพื่อให้โมเดลเข้าใจได้ง่ายขึ้น")
    st.subheader("✅ 3.การเลือกฟีเจอร์ที่สำคัญ (Feature Selection)")
    st.write("- ใช้ Correlation Matrix เพื่อตรวจสอบว่าฟีเจอร์ไหนมีความสัมพันธ์สูงกับราคาบ้าน และลบฟีเจอร์ที่ไม่เกี่ยวข้องออก")
    st.write("- ลบฟีเจอร์ที่ไม่จำเป็นออกเพื่อลดความซับซ้อนของโมเดล")
    st.subheader("✅ 4.การสร้างฟีเจอร์ใหม่ (Feature Engineering)")
    st.write("เพิ่มฟีเจอร์ที่ช่วยให้โมเดลเรียนรู้ได้ดีขึ้น เช่น:")
    st.write("df_filtered[bathroom_ratio] = df_filtered[bathrooms] / (df_filtered[bedrooms] + 1)")
    st.write("df_filtered[area_per_story] = df_filtered[area] / df_filtered[stories]")
    st.write("✅ เป้าหมาย: เพิ่มข้อมูลเชิงลึกให้กับโมเดล")
    st.subheader("✅ 5.การแบ่งชุดข้อมูล (Train-Test Split)")
    st.write("เพื่อให้โมเดลสามารถเรียนรู้และทดสอบได้อย่างมีประสิทธิภาพ จะต้องแบ่งข้อมูลออกเป็น 2 ส่วน:")
    st.write("from sklearn.model_selection import train_test_split")
    st.write("X = df_filtered.drop(columns=[price])  # ฟีเจอร์ที่ใช้ Train")
    st.write("y = df_filtered[price]  # ตัวแปรเป้าหมาย")
    st.write("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    st.write("✅ เป้าหมาย: ใช้ 80% ของข้อมูลสำหรับ Train โมเดล และ 20% สำหรับทดสอบ")
    st.subheader("✅ 6.การทำ Feature Scaling")
    st.write("ฟีเจอร์ที่มีช่วงค่าต่างกันมาก เช่น Area, Bedrooms, Bathrooms ควรทำให้มีขนาดใกล้เคียงกันเพื่อให้โมเดลเรียนรู้ได้ดีขึ้น")
    st.write("from sklearn.preprocessing import StandardScaler")
    st.write("scaler = StandardScaler()")
    st.write("X_train_scaled = scaler.fit_transform(X_train)")
    st.write("X_test_scaled = scaler.transform(X_test)")
    st.write("✅ เป้าหมาย: ป้องกันไม่ให้โมเดลให้ความสำคัญกับฟีเจอร์ที่มีค่ามากเกินไป")



    st.title("Neural Network")
    st.write("- ในโมเดล Neural Network สำหรับพยากรณ์ราคารถยนต์ มีการเตรียมข้อมูลและเลือกอัลกอริทึมที่เหมาะสม โดยกระบวนการมีดังนี้:")
    st.subheader("✅ 1. การโหลดและทำความสะอาดข้อมูล")
    st.write("- ใช้ชุดข้อมูลที่มีรายละเอียดของรถยนต์ เช่น ยี่ห้อรถ, รุ่น, ปีที่ผลิต, ระยะทางที่ขับไปแล้ว (km), ขนาดเครื่องยนต์ (cc), จำนวนที่นั่ง, ประเภทเชื้อเพลิง, ระบบเกียร์ เป็นต้น")
    st.write("- กำจัดค่าข้อมูลที่หายไป (Missing Values) และข้อมูลที่ไม่จำเป็น (Irrelevant Features)")
    st.write("จากการตรวจสอบข้อมูล หากพบว่า มีค่าว่าง จะต้องกำจัดหรือเติมค่าด้วยวิธีการที่เหมาะสม")
    st.write("🔹 วิธีเติมค่าที่หายไป:")
    st.write("- สำหรับข้อมูลตัวเลข → ใช้ค่าเฉลี่ย (Mean Imputation)")
    st.write("- สำหรับข้อมูลประเภทข้อความ → ใช้ค่าที่พบบ่อย (Mode Imputation)")
    st.write("✅ เป้าหมาย: ป้องกันไม่ให้ข้อมูลที่หายไปส่งผลต่อการทำงานของโมเดล")
    st.subheader("✅ 2. การแปลงข้อมูลประเภทข้อความเป็นตัวเลข (Categorical Encoding)")
    st.write("เนื่องจาก Neural Network ต้องการอินพุตที่เป็นตัวเลขทั้งหมด เราจึงต้องแปลงค่าที่เป็นข้อความ เช่น Fuel Type, Transmission, Car Brand")
    st.write("- แปลงค่าที่เป็นตัวอักษร เช่น fuel_type, transmission, car_company, car_model ให้เป็นค่าตัวเลข โดยใช้ One-Hot Encoding")
    st.write("✅ เป้าหมาย: ทำให้ข้อมูลอยู่ในรูปแบบที่ Neural Network สามารถเรียนรู้ได้")
    st.subheader("✅ 3. การลบค่าที่ผิดปกติ (Removing Outliers)")
    st.write("บางข้อมูลอาจมีค่า Outliers เช่น:")
    st.write("- ระยะทาง (kms_driven) ต่ำเกินไปหรือสูงเกินจริง")
    st.write("- ขนาดเครื่องยนต์ (engine_cc) เป็น 0 cc ซึ่งเป็นไปไม่ได้")
    st.write("- ราคาที่ต่ำกว่าหรือสูงกว่าปกติอย่างมาก")
    st.subheader("📌 ใช้ IQR (Interquartile Range) เพื่อลบค่าที่ผิดปกติ:")
    st.write("//คำนวณ IQR")
    st.write("Q1 = df['engine_cc'].quantile(0.25)")
    st.write("Q3 = df['engine_cc'].quantile(0.75)")
    st.write("IQR = Q3 - Q1")
    st.write("//ลบข้อมูลที่อยู่นอกช่วงปกติ")
    st.write("df = df[~((df['engine_cc'] < (Q1 - 1.5 * IQR)) | (df['engine_cc'] > (Q3 + 1.5 * IQR)))]")
    st.write("✅ เป้าหมาย: กำจัดข้อมูลที่ผิดปกติเพื่อให้โมเดลเรียนรู้จากข้อมูลที่มีคุณภาพ")
    st.subheader("✅ 4.การเลือกฟีเจอร์ที่สำคัญ (Feature Selection)")
    st.write("ใช้ Correlation Matrix เพื่อเลือกเฉพาะฟีเจอร์ที่มีผลต่อราคามากที่สุด")
    st.write("import seaborn as sns")
    st.write("import matplotlib.pyplot as plt")
    st.write("plt.figure(figsize=(10, 6))")
    st.write("sns.heatmap(df.corr(), annot=True, cmap=coolwarm)")
    st.write("plt.show()")
    st.subheader("📌 ลบฟีเจอร์ที่มี Correlation ต่ำออก:")
    st.write("columns_to_drop = [car_model_name, car_model_details]  # ฟีเจอร์ที่ไม่เกี่ยวข้อง")
    st.write("df_filtered = df.drop(columns=columns_to_drop)")
    st.write("✅ เป้าหมาย: ลดจำนวนฟีเจอร์ที่ไม่จำเป็นเพื่อลด Overfitting")
    st.subheader("✅ 6.การแบ่งข้อมูล Train-Test (Train-Test Split)")
    st.write("โมเดลต้องได้รับการฝึกบนชุดข้อมูลหนึ่ง (Train Set) และทดสอบบนอีกชุดหนึ่ง (Test Set)")
    st.write("from sklearn.model_selection import train_test_split")
    st.write("X = df_filtered.drop(columns=[car_price])  # ฟีเจอร์ที่ใช้ Train")
    st.write("y = df_filtered[car_price]  # ตัวแปรเป้าหมาย")
    st.write("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    st.write("✅ เป้าหมาย: ให้โมเดลสามารถเรียนรู้จากข้อมูล Train และทดสอบประสิทธิภาพในข้อมูล Test")
    st.subheader("✅ 7. การทำ Feature Scaling (Normalization & Standardization)")
    st.write("-Neural Network มีความไวต่อค่าฟีเจอร์ที่มีช่วงกว้างมาก เช่น ราคารถ, ระยะทางที่ใช้ไป (kms_driven), ขนาดเครื่องยนต์ (engine_cc) จำเป็นต้องปรับค่าฟีเจอร์ให้อยู่ในช่วงที่เหมาะสม")
    st.write("-ใช้ StandardScaler เพื่อปรับข้อมูลให้อยู่ในช่วงที่เหมาะสม")
    st.write("from sklearn.preprocessing import StandardScaler")
    st.write("scaler = StandardScaler()")
    st.write("X_train_scaled = scaler.fit_transform(X_train)")
    st.write("X_test_scaled = scaler.transform(X_test)")
    st.write("✅ เป้าหมาย: ทำให้โมเดลเรียนรู้ได้เร็วขึ้นและลด Bias ที่เกิดจากค่าตัวเลขที่ต่างกันมาก")
    


with tabs[1]:
    st.title("Detail of DataSet(ML)")
    st.header("ที่มาของ Dataset")
    st.write("Housing Prices Dataset เป็นชุดข้อมูลที่ใช้สำหรับวิเคราะห์และพยากรณ์ราคาบ้าน โดยรวบรวมข้อมูลเกี่ยวกับลักษณะของที่พักอาศัยและปัจจัยที่มีผลต่อราคา")
    st.write("อาทิเช่น ขนาดพื้นที่ จำนวนห้องนอน จำนวนห้องน้ำ และอื่น ๆ")
    st.markdown("🔗 แหล่งที่มา: [Kaggle - Housing Prices Dataset](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset)")
    st.subheader("จุดประสงค์ของ Dataset")
    st.write("- ใช้สำหรับการสร้างแบบจำลอง Machine Learning เพื่อพยากรณ์ราคาที่พักอาศัย")
    st.write("- ศึกษาความสัมพันธ์ระหว่างปัจจัยต่าง ๆ กับราคาบ้าน")
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
    st.header("ที่มาของ Dataset")
    st.write("Old Car Price Prediction Dataset เป็นชุดข้อมูลที่ใช้สำหรับวิเคราะห์และพยากรณ์ราคาขายของรถยนต์มือสอง โดยรวบรวมข้อมูลเกี่ยวกับ ลักษณะของรถยนต์, ปีที่ผลิต,")
    st.write("ระยะทางที่ขับไปแล้ว, ขนาดเครื่องยนต์, ระบบเกียร์, ประเภทเชื้อเพลิง, ยี่ห้อรถยนต์ และปัจจัยอื่นๆ ที่มีผลต่อราคาขาย")
    st.markdown("🔗 แหล่งที่มา: [Kaggle - Old Car Price Prediction Dataset](https://www.kaggle.com/datasets/milanvaddoriya/old-car-price-prediction/code)")
    st.subheader("📌 จุดประสงค์ของ Dataset")
    st.write("- ใช้สำหรับการสร้างแบบจำลอง Machine Learning และ Neural Network เพื่อพยากรณ์ราคารถยนต์มือสอง")
    st.write("- ศึกษาความสัมพันธ์ระหว่างปัจจัยต่าง ๆ กับราคารถยนต์ เช่น ปีที่ผลิต ระยะทางที่ขับไปแล้ว และยี่ห้อรถยนต์")
    st.write("- ช่วยให้ผู้ซื้อและผู้ขายสามารถประเมินราคาของรถยนต์มือสองได้แม่นยำขึ้น")
    st.header("Detail of DataSet(NN)")
    st.write("ชุดข้อมูลนี้ประกอบด้วย 5512 แถว และ 11 คอลัมน์ ซึ่งแสดงรายละเอียดของแต่ละคุณลักษณะได้ดังนี้:")
    df = pd.read_csv("car_price.csv")
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
        <tr>
            <td>car_company_name</td>
            <td>ยี่ห้อของรถยนต์</td>
            <td>object</td>
        </tr>
        <tr>
            <td>car_model_name</td>
            <td>รุ่นของรถยนต์</td>
            <td>object</td>
        </tr>
        <tr>
            <td>manufacture</td>
            <td>ปีที่ผลิต</td>
            <td>int</td>
        </tr>
        <tr>
            <td>fuel_type</td>
            <td>ชนิดของเชื้อเพลิง (Petrol, Diesel, CNG)</td>
            <td>object</td>
        </tr>
        <tr>
            <td>transmission</td>
            <td>ระบบเกียร์ (Manual, Automatic)</td>
            <td>object</td>
        </tr>
        <tr>
            <td>kms_deriven_numeric</td>
            <td>ระยะทางที่ขับไปแล้ว (กม.)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>engine_cc_numeric</td>
            <td>ขนาดเครื่องยนต์ (cc)</td>
            <td>int</td>
        </tr>
        <tr>
            <td>seats_numeric</td>
            <td>จำนวนที่นั่ง</td>
            <td>int</td>
        </tr>
        <tr>
            <td>ownership_numeric</td>
            <td>ลำดับเจ้าของรถ</td>
            <td>int</td>
        </tr>
        <tr>
            <td>Price</td>
            <td>ราคารถยนต์ที่ต้องการพยากรณ์ (เป้าหมาย)</td>
            <td>float</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)



with tabs[3]:
    st.title("🏗 Development Model (กระบวนการสร้างโมเดล)")
    st.title("Machine Learning")
    st.subheader("1️⃣ การเลือกอัลกอริทึม (Algorithm Selection)")
    st.write("เราได้ทดลองใช้อัลกอริทึมหลายแบบเพื่อหาวิธีที่เหมาะสมที่สุดในการพยากรณ์ราคาบ้าน ได้แก่:")
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
            <th>อัลกอริทึมที่ทดลอง</th>
            <th>ข้อดี</th>
            <th>ข้อเสีย</th>
        </tr>
        <tr>
            <td>Linear Regression</td>
            <td>เรียนง่าย, คำนวณเร็ว</td>
            <td>ไม่สามารถจับความสัมพันธ์ที่ซับซ้อนได้</td>
        </tr>
        <tr>
            <td>Decision Tree</td>
            <td>เข้าใจง่าย, รองรับข้อมูลเชิงหมวดหมู่</td>
            <td>Overfitting ได้ง่าย</td>
        </tr>
        <tr>
            <td>Random Forest</td>
            <td>ลด Overfitting ได้ดี, แม่นยำกว่าต้นไม้เดี่ยว</td>
            <td>ใช้ทรัพยากรสูง, ช้ากว่า Decision Tree</td>
        </tr>
        <tr>
            <td>Gradient Boosting</td>
            <td>เรียนรู้จากข้อผิดพลาดของตนได้, ปรับแต่งพารามิเตอร์ได้ดี</td>
            <td>เพราะซับซ้อนกว่า Random Forest</td>
        </tr>
        <tr>
            <td>XGBoost (เลือกใช้)</td>
            <td>ประสิทธิภาพสูง, รองรับฟีเจอร์ที่หลากหลาย, ป้องกัน Overfitting</td>
            <td>ต้องปรับพารามิเตอร์ให้เหมาะสม</td>
        </tr>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.write("✅ ผลลัพธ์: หลังจากการทดลอง พบว่า XGBoost ให้ผลลัพธ์ดีที่สุด โดยมี Mean Absolute Error (MAE) ต่ำสุด และสามารถจัดการกับข้อมูลที่ซับซ้อนได้ดี")

    st.subheader("2️⃣ การตั้งค่า Hyperparameter (Hyperparameter Tuning)")
    st.write("หลังจากเลือก XGBoost เป็นโมเดลหลัก เราได้ทำ Hyperparameter Tuning เพื่อให้โมเดลมีประสิทธิภาพสูงสุด โดยใช้เทคนิค:")
    st.write("- Randomized Search CV → ทดลองค่าพารามิเตอร์แบบสุ่มเพื่อหาเซ็ตค่าที่ดีที่สุด")
    st.write("- Grid Search CV → ทดลองค่าพารามิเตอร์แบบละเอียดเพื่อปรับปรุงโมเดลให้ดียิ่งขึ้น")
    st.subheader("🔹 ค่าพารามิเตอร์ที่ดีที่สุดที่ใช้ในโมเดลนี้:")
    st.write("from xgboost import XGBRegressor")
    st.write("xgb_model = XGBRegressor(")
    st.write("   booster= gbtree ,        # ใช้ Gradient Boosted Trees")
    st.write("   n_estimators=400,        # จำนวนต้นไม้ที่ใช้")
    st.write("   learning_rate=0.01,      # อัตราการเรียนรู้")
    st.write("   max_depth=5,             # ความลึกของต้นไม้")
    st.write("   colsample_bytree=0.6,    # ใช้ฟีเจอร์สุ่ม 60% ในแต่ละต้นไม้")
    st.write("   subsample=0.8,           # ใช้ข้อมูลสุ่ม 80% ในแต่ละรอบ")
    st.write("   random_state=42          # ค่าคงที่เพื่อให้ผลลัพธ์ทำซ้ำได้")
    st.write(")")
    st.subheader("📌 ความสำคัญของพารามิเตอร์หลัก:")
    st.write("- booster= gbtree  → ใช้ต้นไม้ตัดสินใจแบบ Gradient Boosting")
    st.write("- n_estimators=400 → จำนวนต้นไม้ที่ใช้ในการฝึกโมเดล")
    st.write("- learning_rate=0.01 → ทำให้โมเดลเรียนรู้ช้าแต่ลด Overfitting ได้")
    st.write("- max_depth=5 → จำกัดความลึกของต้นไม้เพื่อลด Overfitting")
    st.write("- subsample=0.8 → ใช้ข้อมูลสุ่ม 80% ในแต่ละรอบการฝึก")
    st.write("- colsample_bytree=0.6 → ใช้ฟีเจอร์เพียง 60% ในแต่ละต้นไม้เพื่อลดความซ้ำซ้อน")
    st.subheader("3️⃣ การฝึกโมเดล (Training Process)")
    st.write("หลังจากตั้งค่าพารามิเตอร์แล้ว โมเดลจะถูกฝึกด้วยชุดข้อมูลที่ผ่านการ Feature Engineering และ Scaling โดย:")
    st.write("- ใช้ Cross-Validation (K-Fold = 5) เพื่อทดสอบความสามารถของโมเดลในชุดข้อมูลที่แตกต่างกัน")
    st.write("- ใช้ Early Stopping เพื่อลดปัญหา Overfitting โดยหยุดการฝึกเมื่อค่า Loss ไม่ลดลง")
    st.subheader("🔹 โค้ดสำหรับฝึกโมเดล:")
    st.write("from sklearn.model_selection import train_test_split")
    st.subheader("แบ่งข้อมูลเป็น Train และ Test")
    st.write("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)")
    st.subheader("ฝึกโมเดล XGBoost")
    st.write("xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=True)")
    st.subheader("4️⃣ การวัดผลโมเดล (Model Evaluation)")
    st.write("หลังจากฝึกโมเดลแล้ว จะต้องตรวจสอบว่ามีความแม่นยำเพียงใด โดยใช้: ✅ Mean Absolute Error (MAE) → วัดค่าความคลาดเคลื่อนโดยเฉลี่ย")
    st.write("✅ Root Mean Squared Error (RMSE) → วัดค่าคลาดเคลื่อนที่ให้ความสำคัญกับค่าผิดพลาดมาก")
    st.write("✅ R-Squared (R² Score) → วัดว่าข้อมูลที่โมเดลเรียนรู้สามารถอธิบายได้ดีเพียงใด")
    st.subheader("🔹 โค้ดวัดผลโมเดล:")
    st.write("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score")
    st.subheader("ทำนายราคาบ้าน")
    st.write("y_pred = xgb_model.predict(X_test)")
    st.subheader("คำนวณค่าความคลาดเคลื่อน")
    st.write("mae = mean_absolute_error(y_test, y_pred)")
    st.write("rmse = mean_squared_error(y_test, y_pred, squared=False)")
    st.write("r2 = r2_score(y_test, y_pred)")
    st.write("print(f 📊 MAE: {mae:.2f} )")
    st.write("print(f 📊 RMSE: {rmse:.2f} )")
    st.write("print(f 📊 R² Score: {r2:.4f} )")
    st.subheader("5️⃣ การบันทึกโมเดล (Model Deployment)")
    st.write("เมื่อได้โมเดลที่ดีที่สุดแล้ว จะทำการบันทึกโมเดลเพื่อใช้ในระบบจริง โดย:")
    st.write("- บันทึกโมเดลเป็น xgboost_best_model.pkl เพื่อให้สามารถโหลดไปใช้งานได้ง่าย")
    st.write("- ใช้ joblib ในการบันทึกโมเดลเพื่อให้โหลดได้เร็วขึ้น")
    st.subheader("🔹 โค้ดบันทึกโมเดล:")
    st.write("import joblib")
    st.write("บันทึกโมเดล")
    st.write("joblib.dump(xgb_model,  xgboost_best_model.pkl )")
    st.write("print( ✅ โมเดลถูกบันทึกเรียบร้อยแล้ว! )")
    st.subheader("📌 สรุป")
    st.write("1️⃣ ได้ทดลองใช้อัลกอริทึมหลายแบบ เช่น Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost")
    st.write("2️⃣ XGBoost เป็นโมเดลที่ดีที่สุด เนื่องจากมีค่า MAE ต่ำสุด และสามารถจัดการกับฟีเจอร์ที่ซับซ้อนได้ดี")
    st.write("3️⃣ ปรับแต่งพารามิเตอร์ด้วย Grid Search และ Cross Validation เพื่อลด Overfitting และเพิ่มความแม่นยำ")
    st.write("4️⃣ ใช้ Early Stopping และ Feature Selection เพื่อให้โมเดลทำงานได้เสถียรและไม่ Overfit")
    st.write("5️⃣ โมเดลสุดท้ายถูกบันทึกเป็น xgboost_best_model.pkl เพื่อนำไปใช้งานจริงใน Web Application (Streamlit)")
    



    
    st.title("Neural Network")
    st.subheader("✅ 1. การออกแบบโครงสร้าง Neural Network")
    st.write("โมเดลที่ใช้เป็น Fully Connected Neural Network (Multilayer Perceptron - MLP) ซึ่งมีโครงสร้างดังนี้:")
    st.write("-Input Layer → Hidden Layer 1 (ReLU) → Hidden Layer 2 (ReLU) → Hidden Layer 3 (ReLU) → Output Layer (Linear)")
    st.write("Input Layer: จำนวน Neurons เท่ากับจำนวนฟีเจอร์ที่เลือกใช้")
    st.write("Hidden Layers:")
    st.write("- Layer 1: 64 Neurons + ReLU Activation")
    st.write("- Layer 2: 32 Neurons + ReLU Activation")
    st.write("- Layer 3: 16 Neurons + ReLU Activation")
    st.write("Output Layer: 1 Neuron (ทำนายราคารถยนต์) + Linear Activation")
    st.subheader("✅ 2. การเลือก Loss Function และ Optimizer")
    st.write("- ใช้ Mean Squared Error (MSE) เป็น Loss Function เนื่องจากเป็นปัญหาการพยากรณ์แบบ Regression")
    st.write("ใช้ Adam Optimizer ซึ่งเป็นอัลกอริทึมที่ช่วยให้โมเดลเรียนรู้ได้เร็วขึ้นและลดค่า Loss ได้ดี")
    st.subheader("✅ 3. การแบ่งข้อมูล Train/Test และการเทรนโมเดล")
    st.write("- แบ่งข้อมูลเป็น 80% Train, 20% Test")
    st.write("- ใช้ EarlyStopping เพื่อลดปัญหา Overfitting โดยหยุดการเทรนเมื่อ Loss ไม่ลดลง")
    st.write("- เทรนโมเดลด้วย Batch Size 32 และ Epochs 100 (แต่ใช้ EarlyStopping เพื่อลดจำนวน Epochs ที่ไม่จำเป็น)")
    st.subheader("✅ 4. การประเมินผลโมเดล")
    st.write("- ใช้ค่า Mean Absolute Error (MAE) และ Root Mean Squared Error (RMSE) วัดผล")
    st.write("- ทำ Cross Validation และตรวจสอบว่าโมเดลสามารถ Generalize ได้ดีหรือไม่")
    st.subheader("✅ 5. การบันทึกโมเดล")
    st.write("- โมเดลที่ได้ถูกบันทึกเป็นไฟล์ car_price_model.keras เพื่อให้สามารถโหลดไปใช้ภายหลังได้โดยตรง")
    st.write("- การแปลงข้อมูลก่อนเข้าโมเดลถูกบันทึกเป็น column_transformer.pkl เพื่อให้สามารถแปลงข้อมูลใหม่ได้เหมือนเดิม")
    st.subheader("📌 สรุป")
    st.write("- โมเดลที่ใช้เป็น Fully Connected Neural Network (MLP) ที่ผ่านการปรับแต่ง Hyperparameter และเลือกฟีเจอร์ที่เหมาะสม")
    st.write("- ใช้การ Feature Scaling, One-Hot Encoding, และ Standardization เพื่อให้โมเดลสามารถเรียนรู้ข้อมูลได้ดีขึ้น")
    st.write("- การพัฒนาโมเดลเน้นการลด Overfitting ด้วย EarlyStopping และ Regularization")
    st.write("- โมเดลสามารถพยากรณ์ราคารถยนต์ได้อย่างแม่นยำและรองรับการใช้งานผ่าน Web Application (Streamlit Demo)")





with tabs[4]:
    st.title("🧠Theory of ML Model")
    st.header("การทดลองใช้อัลกอริทึมสำหรับพยากรณ์ราคาบ้าน")
    st.write("จากการพัฒนาโมเดล เราได้ทดลองใช้อัลกอริทึมหลายชนิดเพื่อหาวิธีที่มีประสิทธิภาพที่สุดในการพยากรณ์ราคาบ้าน โดยมีรายละเอียดของอัลกอริทึมที่ใช้ ดังนี้:")
    st.subheader("✅ 1. Linear Regression)")
    st.write("- อัลกอริทึมพื้นฐานที่ใช้เส้นตรงเป็นตัวแบบในการพยากรณ์")
    st.write("- ได้ค่า Mean Absolute Error (MAE) ค่อนข้างสูง แสดงว่าโมเดลมีความคลาดเคลื่อนมาก")
    st.write("- เมื่อเพิ่มการ Standard Scaling ทำให้ค่าคลาดเคลื่อนลดลงเล็กน้อย แต่ยังไม่น่าพอใจ")
    st.subheader("✅ 2. Decision Tree")
    st.write("- ใช้โครงสร้างต้นไม้ตัดสินใจในการพยากรณ์ราคา")
    st.write("- แม้ว่าจะสามารถจับความสัมพันธ์แบบไม่เชิงเส้นได้ดี แต่โมเดลมีปัญหาการ Overfitting")
    st.write("- ค่า MAE ต่ำในชุด Train แต่สูงขึ้นในชุด Test ซึ่งหมายถึงโมเดลไม่สามารถ Generalize ได้ดี")
    st.subheader("✅ 3. Random Forest")
    st.write("- ใช้ต้นไม้ตัดสินใจหลายต้นเพื่อรวมผลการพยากรณ์ให้แม่นยำขึ้น")
    st.write("- ทดสอบจำนวนต้นไม้ที่แตกต่างกัน (50, 100, 200) และความลึกของต้นไม้ (10, 20, None)")
    st.write("- พบว่าโมเดลสามารถลดค่าคลาดเคลื่อนได้มากกว่า Decision Tree แต่ยังมีความผันผวนอยู่")
    st.subheader("✅ 4. Gradient Boosting")
    st.write("เป็นเทคนิคที่เรียนรู้ข้อผิดพลาดของโมเดลก่อนหน้าแล้วปรับปรุงโมเดลใหม่")
    st.write("- ทดลองค่าพารามิเตอร์ต่าง ๆ เช่น Learning Rate และ Max Depth")
    st.write("- พบว่าโมเดลสามารถลด MAE ลงได้ดีขึ้นเมื่อใช้พารามิเตอร์ที่เหมาะสม")
    st.subheader("✅ 5. XGBoost")
    st.write("- โมเดลที่พัฒนาต่อจาก Gradient Boosting โดยมีการเพิ่มประสิทธิภาพในการคำนวณและลด Overfitting")
    st.write("- ทดลองพารามิเตอร์หลายชุดและใช้ Randomized Search CV เพื่อค้นหาค่าที่ดีที่สุด")
    st.write("- ใช้การเลือกฟีเจอร์อัตโนมัติเพื่อลดจำนวนฟีเจอร์ที่ไม่จำเป็นออก")
    st.subheader("✅ 6. LightGBM")
    st.write("- เป็นอัลกอริทึมที่คล้าย XGBoost แต่มีความเร็วสูงกว่า")
    st.write("- ทดลองเทรนโมเดลและพบว่าค่า MAE ใกล้เคียงกับ XGBoost แต่ไม่ดีกว่า")
    st.subheader("✅ 7. Stacking Model")
    st.write("- ใช้โมเดลหลายตัว (Random Forest, Gradient Boosting, XGBoost) มารวมกันเป็น Super Model")
    st.write("- โมเดลสุดท้ายที่ใช้พยากรณ์เป็น XGBoost")
    st.write("- แม้ว่าค่า MAE จะลดลงเล็กน้อย แต่ใช้ทรัพยากรคอมพิวเตอร์สูงมาก")
    st.header("🔥 เหตุผลที่เลือก โมเดล XGBoost")
    st.subheader("1.ค่า MAE ต่ำสุด")
    st.write("- หลังจากปรับพารามิเตอร์ XGBoost พบว่าค่า Mean Absolute Error (MAE) ต่ำกว่าทุกโมเดลที่ทดลอง")
    st.write("- ใช้ Randomized Search CV เพื่อหา Hyperparameter ที่ดีที่สุด เช่น n_estimators, learning_rate, max_depth, colsample_bytree, และ subsample")
    st.subheader("2.สามารถเลือกฟีเจอร์ที่สำคัญได้")
    st.write("- ใช้ Feature Selection เพื่อลบฟีเจอร์ที่ไม่สัมพันธ์กับราคาบ้าน")
    st.write("- ลดความซับซ้อนของโมเดล ทำให้ประสิทธิภาพดีขึ้น")
    st.subheader("3.รองรับการเพิ่มฟีเจอร์ใหม่")
    st.write("- สร้างฟีเจอร์ใหม่ เช่น bathroom_ratio, area_per_story, room_density ซึ่งช่วยเพิ่มความแม่นยำของโมเดล")
    st.subheader("4.รองรับทั้ง Booster Type: gbtree และ dart")
    st.write("- ทดสอบโมเดล XGBoost สองแบบ gbtree และ dart")
    st.write("- พบว่า gbtree ทำงานได้เร็วกว่าและให้ผลลัพธ์ที่ดีในกรณีนี้")
    st.subheader("5.การใช้ Scaling ช่วยปรับปรุงผลลัพธ์")
    st.write("- ใช้ StandardScaler ช่วยให้โมเดลสามารถทำงานกับข้อมูลที่มีค่าต่างกันมากได้ดีขึ้น")
    st.write("- ลดปัญหาการ Bias ของฟีเจอร์ที่มีช่วงค่าต่างกัน")
    st.subheader("6.ทดสอบโมเดลในหลายชุดข้อมูล")
    st.write("- ใช้ Cross Validation (5-Fold) ตรวจสอบว่าโมเดลสามารถ Generalize ได้ดีหรือไม่")
    st.write("- พบว่าค่า MAE มีค่าคงที่ในแต่ละ Fold แสดงถึงความเสถียรของโมเดล")
    st.subheader("📌 สรุป")
    st.write("- ได้ทดลองใช้อัลกอริทึมทั้งหมด 7 แบบ (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, Stacking)")
    st.write("- หลังจากทดลองพารามิเตอร์และเลือกฟีเจอร์ที่ดีที่สุด พบว่า XGBoost หลัง Advanced Tuning ให้ผลลัพธ์ที่ดีที่สุด")
    st.write("- โมเดลที่ใช้ในโปรเจคเป็น XGBoost (Booster: gbtree) ที่มีการปรับพารามิเตอร์มาอย่างดี")
    st.write("- โมเดลสามารถ รองรับข้อมูลใหม่, คำนวณได้เร็ว, มีค่า MAE ต่ำ และใช้งานได้เสถียร")



with tabs[5]:  
    st.title("Theory of NN Model")
    st.header("การทดลองใช้อัลกอริทึมสำหรับพยากรณ์ราคารถยนต์")
    st.write("เราได้ทดลองใช้ Neural Network โดยใช้โครงข่ายประสาทเทียม (ANN) แบบ Fully Connected Layers (Dense Layers) โดยโมเดลที่พัฒนาเป็น Regression Model")
    st.write("ซึ่งทำงานโดยการคาดการณ์ค่าตัวเลขที่ต่อเนื่อง เช่น ราคา โดยใช้ฟังก์ชันการสูญเสียที่เหมาะสมกับปัญหาการพยากรณ์เช่น Mean Squared Error (MSE) หรือ Mean Absolute Error (MAE)")
    st.subheader("1.1 โครงสร้างของ Neural Network ที่ใช้")
    st.write("โมเดลที่ถูกพัฒนาประกอบด้วย:")
    st.write("- 3 ชั้นซ่อน (Hidden Layers)")
    st.write("- ใช้ Activation Function ที่เหมาะสมกับการพยากรณ์ค่า เช่น ReLU และ Linear")
    st.write("- ใช้ Optimizer: Adam ซึ่งช่วยให้โมเดลสามารถเรียนรู้ได้เร็วขึ้น")
    st.write("- Loss Function: Mean Absolute Error (MAE) ใช้วัดความผิดพลาดของการพยากรณ์ราคา")
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
        <th>Layer</th>
        <th>Type</th>
        <th>Output Shape</th>
        <th>Param #</th>
    </tr>
    <tr>
        <td>Dense_6</td>
        <td>Fully Connected (Dense)</td>
        <td>(None, 16)</td>
        <td>4,192</td>
    </tr>
    <tr>
        <td>Dense_7</td>
        <td>Fully Connected (Dense)</td>
        <td>(None, 8)</td>
        <td>136</td>
    </tr>
    <tr>
        <td>Dense_8</td>
        <td>Fully Connected (Dense)</td>
        <td>(None, 1)</td>
        <td>9</td>
    </tr>
</table>
"""

    st.markdown(table_html, unsafe_allow_html=True)
    st.write("Total Parameters: 13,013")

    st.header("🔥เหตุผลในการเลือก Neural Network Regression")
    st.write("จากการทดลองใช้ Neural Network Regression พบว่า:")
    st.write("- Neural Network สามารถจับความสัมพันธ์ที่ซับซ้อนของข้อมูลได้ดี")
    st.write("- มีความสามารถในการเรียนรู้ Feature Interactions ได้ดีกว่าอัลกอริทึมแบบตื้น (Shallow Learning)")
    st.write("- สามารถปรับแต่งโครงสร้างโมเดลให้เหมาะสมกับข้อมูลที่มีจำนวนมากขึ้น")
    st.write("อย่างไรก็ตาม Neural Network ใช้เวลาฝึกสอนนานกว่า XGBoost และอาจต้องการข้อมูลจำนวนมากกว่าเพื่อให้ได้ผลลัพธ์ที่แม่นยำ")
    st.write("ดังนั้น Neural Network Regression จึงเป็นทางเลือกที่เหมาะสมสำหรับการพยากรณ์ราคารถยนต์ในกรณีที่ต้องการให้โมเดลเรียนรู้ความสัมพันธ์ที่ซับซ้อนของข้อมูล")
