import streamlit as st

st.title("📊 Model Explanation (คำอธิบายโมเดล)")

st.header("Machine Learning Model")
st.subheader("Feature Importance")
st.write("โมเดลให้ความสำคัญกับฟีเจอร์ต่าง ๆ ดังนี้:")
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
        <th>Importance Score</th>
    </tr>
    <tr><td>Area (ตร.ม.)</td><td>0.35</td></tr>
    <tr><td>Number of Bedrooms</td><td>0.20</td></tr>
    <tr><td>Number of Bathrooms</td><td>0.15</td></tr>
    <tr><td>Furnishing Status</td><td>0.10</td></tr>
    <tr><td>Parking Spaces</td><td>0.08</td></tr>
    <tr><td>Proximity to Main Road</td><td>0.07</td></tr>
    <tr><td>Prefarea (Prime Location)</td><td>0.05</td></tr>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)
st.subheader("📌 ฟีเจอร์ที่มีผลต่อการพยากรณ์ราคามากที่สุด ได้แก่")
st.write("🔹 ฟีเจอร์ Area และ Number of Bedrooms มีผลมากที่สุดในการพยากรณ์ราคา")


st.header("Neural Network Model")
st.subheader("Feature Importance")
st.write("โมเดลให้ความสำคัญกับฟีเจอร์ต่าง ๆ ดังนี้:")
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
        <th>Importance Score</th>
    </tr>
    <tr>
        <td>Manufacture Year (ปีที่ผลิต)</td>
        <td>0.30</td>
    </tr>
    <tr>
        <td>Kilometers Driven (ระยะทางที่ขับ)</td>
        <td>0.25</td>
    </tr>
    <tr>
        <td>Car Brand (ยี่ห้อรถ)</td>
        <td>0.15</td>
    </tr>
    <tr>
        <td>Car Model (รุ่นรถ)</td>
        <td>0.10</td>
    </tr>
    <tr>
        <td>Fuel Type (ประเภทเชื้อเพลิง)</td>
        <td>0.08</td>
    </tr>
    <tr>
        <td>Transmission (ระบบเกียร์)</td>
        <td>0.07</td>
    </tr>
    <tr>
        <td>Engine Size (ขนาดเครื่องยนต์)</td>
        <td>0.05</td>
    </tr>
</table>
"""

st.markdown(table_html, unsafe_allow_html=True)
st.subheader("📌 ฟีเจอร์ที่มีผลต่อการพยากรณ์ราคามากที่สุด ได้แก่")
st.write("1️⃣ ปีที่ผลิตของรถยนต์ (Manufacture Year) – รถใหม่กว่ามักมีราคาสูงกว่า")
st.write("2️⃣ ระยะทางที่ขับไปแล้ว (Kilometers Driven) – รถที่วิ่งมาน้อยมักมีราคาสูงกว่า")
st.write("3️⃣ ยี่ห้อรถยนต์ (Car Brand) – แบรนด์พรีเมียมมักมีราคาสูงกว่า")







st.subheader("เปรียบเทียบโมเดล Machine Learning กับ Neural Network")
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
        <th>XGBoost (Machine Learning)</th>
        <th>Neural Network (Deep Learning)</th>
    </tr>
    <tr>
        <td>ประเภทของปัญหา</td>
        <td>Regression / Classification</td>
        <td>Regression / Complex Patterns</td>
    </tr>
    <tr>
        <td>ความซับซ้อนของโมเดล</td>
        <td>ต่ำกว่า Neural Network</td>
        <td>ซับซ้อนกว่า แต่เรียนรู้ได้ลึกกว่า</td>
    </tr>
    <tr>
        <td>การเรียนรู้ของโมเดล</td>
        <td>ใช้ Gradient Boosting</td>
        <td>ใช้ Multi-Layer Perceptron (MLP)</td>
    </tr>
    <tr>
        <td>ความสามารถในการเรียนรู้ฟีเจอร์ใหม่</td>
        <td>จำกัดตามที่กำหนด</td>
        <td>สามารถสร้าง Feature เองได้</td>
    </tr>
    <tr>
        <td>ความเร็วในการเทรน</td>
        <td>เร็วกกว่า Neural Network</td>
        <td>ช้ากว่าโดยเฉพาะบน CPU</td>
    </tr>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)
#st.write("💡 XGBoost เหมาะกับปัญหาที่มีโครงสร้างชัดเจน เช่น การพยากรณ์ราคาบ้าน ส่วน Neural Network เหมาะกับปัญหาที่ซับซ้อน")




