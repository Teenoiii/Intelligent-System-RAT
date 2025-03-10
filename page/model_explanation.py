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
st.write("🔹 ฟีเจอร์ Area และ Number of Bedrooms มีผลมากที่สุดในการพยากรณ์ราคา")


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
        <td>ต่ำกว่ากว่า Neural Network</td>
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
        <td>ความเร็วในการทรน</td>
        <td>เร็วกกว่า Neural Network</td>
        <td>ช้ากว่าโดยเฉพาะบน CPU</td>
    </tr>
</table>
"""
st.markdown(table_html, unsafe_allow_html=True)
st.write("💡 XGBoost เหมาะกับปัญหาที่มีโครงสร้างชัดเจน เช่น การพยากรณ์ราคาบ้าน ส่วน Neural Network เหมาะกับปัญหาที่ซับซ้อน")



st.subheader("Neural Network Model")
st.write("ออกแบบโครงสร้าง Neural Network ด้วย Keras และ TensorFlow")
