import streamlit as st

st.title("Welcome to ML & Neural Network Web App")
st.write("เว็บแอปนี้ช่วยให้คุณเข้าใจและทดลองโมเดล Machine Learning และ Neural Network")

# สร้าง Tab Bar
tabs = st.tabs(["Introduction", "what is AI ML DL", "what is dataset", "Preparing data", "Machine Learning", "Neural Network"])

#  แสดงเนื้อหาตามแท็บที่เลือก
with tabs[0]:  
    st.header("Introduction Intellgent System")
    st.write("เว็บไซต์นี้เป็นส่วนหนึ่งของรายวิชา Intellgent System ใช้ในการศึกษาการใช้ Machine Learning และ Neural Network ในการแก้ปัญหาต่าง ๆ")

    st.subheader("🔹 จุดประสงค์ของเว็บไซต์ ")
    st.write("-ให้ความเข้าใจเกี่ยวกับกระบวนการพัฒนาโมเดล AI ตั้งแต่การจัดเตรียมข้อมูลไปจนถึงการนำไปใช้งานจริง")
    st.write("-อธิบายแนวคิดพื้นฐานของ Machine Learning และ Neural Network")
    st.write("-นำเสนอตัวอย่างการทำงานของโมเดล Machine Learning และ Neural Network เพื่อให้เข้าใจอย่างเป็นรูปธรรม")
    st.write("-เปรียบเทียบข้อแตกต่างระหว่างอัลกอริทึมประเภทต่างๆ รวมถึงการนำไปใช้ในสถานการณ์ที่เหมาะสม")

    st.subheader("🔹 เนื้อหาในเว็บไซต์ ")
    st.write("-Introduction อธิบายภาพรวมของเว็บไซต์และวัตถุประสงค์หลัก")
    st.write("-Algorithm & Model Development แนวทางการออกแบบและพัฒนาโมเดล Machine Learning และ Neural Network")
    st.write("-Machine Learning Model  วิเคราะห์การทำงานของโมเดล Machine Learning ในรูปแบบต่างๆ")
    st.write("-Neural Network Model – ศึกษากระบวนการทำงานของโมเดล Neural Network พร้อมตัวอย่างการใช้งาน")

    st.subheader("📌ผู้จัดทำ")
    st.write("นาย รัตน์ คงคารัตน์")
    st.write("รหัสนักศึกษา 6404062636447")

    st.subheader("📌อาจารย์ผู้สอน")
    st.write("ดร.ธรรศฏภณ สุระศักดิ์ (TSR)")
with tabs[1]:  
    st.title("what is AI ML DL")

    st.subheader("🌍 Artificial Intelligence (AI) - ปัญญาประดิษฐ์")
    st.write("AI คือแนวคิดที่กว้างที่สุด หมายถึงความสามารถของคอมพิวเตอร์ในการ คิด วิเคราะห์ และตัดสินใจ คล้ายมนุษย์ 🧠✨"
    " เช่น ระบบแชทบอท 🗣️, การจดจำใบหน้า 📸 หรือรถยนต์ไร้คนขับ 🚗💨")

    st.subheader("📊 Machine Learning (ML) - การเรียนรู้ของเครื่อง")
    st.write("ML เป็นแขนงหนึ่งของ AI ที่ใช้ อัลกอริธึม เพื่อให้คอมพิวเตอร์เรียนรู้จากข้อมูลโดยอัตโนมัติ 🔄 ไม่ต้องตั้งโปรแกรมแบบตายตัว 📈 เช่น")
    st.write("✅ ระบบแนะนำหนัง 🎬 (Netflix, YouTube)")
    st.write("✅ การตรวจจับสแปม ✉️")

    st.subheader("🧠 Deep Learning (DL) - การเรียนรู้เชิงลึก")
    st.write("DL เป็นส่วนหนึ่งของ ML ที่ใช้ Neural Networks แบบหลายชั้น 🕸️ เพื่อให้คอมพิวเตอร์สามารถวิเคราะห์ข้อมูลที่ซับซ้อน เช่น")
    st.write("🚀 การจดจำใบหน้า 😃")
    st.write("🎨 AI วาดภาพจากข้อความ")
    st.write("🔊 แปลงเสียงพูดเป็นข้อความ")
    st.image("images/ai_image.jpg", width=500)

with tabs[2]:  
    st.title("what is dataset")

    st.subheader("🔍 Dataset (ชุดข้อมูล) คืออะไร?")
    st.write("Dataset หรือ ชุดข้อมูล คือ กลุ่มของข้อมูลที่ถูกเก็บรวบรวม ไว้ในรูปแบบต่างๆ เพื่อใช้สำหรับการเรียนรู้ของเครื่อง (Machine Learning) 🧠✨")
    
    st.subheader("🔹 ประเภทของ Dataset")
    st.write("🔹 Structured Data (ข้อมูลเชิงโครงสร้าง) 🏛️ – มีรูปแบบชัดเจน เช่น ตารางข้อมูล 📊 (CSV, Excel)")
    st.write("🔹 Unstructured Data (ข้อมูลไร้โครงสร้าง) 📸 – ไม่มีรูปแบบชัดเจน เช่น รูปภาพ, วิดีโอ, ข้อความ")

    st.subheader("🛠 Dataset ใช้ทำอะไร?")
    st.write("✅ ใช้ฝึกโมเดล AI และ Machine Learning 🤖")
    st.write("✅ ใช้วิเคราะห์แนวโน้มข้อมูล 📈")
    st.write("✅ ใช้พัฒนาแอปพลิเคชัน เช่น ระบบแนะนำสินค้า 🛒")

with tabs[3]:
    st.title("Preparing data")
    st.subheader("📌 ทำไมต้องเตรียมข้อมูล?")
    st.write("ก่อนนำข้อมูลไปใช้ใน Machine Learning หรือ Neural Networks 🧠 เราต้องทำให้ข้อมูลสะอาดและอยู่ในรูปแบบที่เหมาะสม 📊 เพราะข้อมูลที่ไม่สมบูรณ์ อาจทำให้โมเดลเรียนรู้ผิดพลาด ❌")
    
    st.subheader("🛠 ขั้นตอนการเตรียมข้อมูล")
    st.write("✅ 1. การทำความสะอาดข้อมูล (Data Cleaning) 🧹ลบข้อมูลที่ซ้ำกัน 🔄, แก้ไขค่าที่หายไป ❓ และจัดรูปแบบให้ถูกต้อง")
    st.write("✅ 2. การแปลงข้อมูล (Data Transformation) 🔄แปลงข้อมูลที่เป็นข้อความ 📜 หรือ หมวดหมู่ เป็นตัวเลข 🔢 เพื่อให้โมเดลเข้าใจ")
    st.write("✅ 3. การทำให้ข้อมูลสมดุล (Balancing Data) ⚖️ถ้ามีข้อมูลบางประเภทมากเกินไป อาจต้องใช้ Oversampling หรือ Undersampling")
    st.write("✅ 4. การทำให้ข้อมูลอยู่ในช่วงที่เหมาะสม (Normalization / Standardization) 📏ช่วยให้ค่าตัวเลขมีสเกลที่เหมาะสม ลดผลกระทบของค่าที่แตกต่างกันมาก")
    st.write("✅ 5. การแบ่งข้อมูล (Train-Test Split) ✂️แบ่งข้อมูลออกเป็น ชุดฝึก (Training Set) และ ชุดทดสอบ (Test Set) เพื่อประเมินประสิทธิภาพของโมเดล")

with tabs[4]:
    st.title("Machine Learning")

    st.subheader("🧠 แนวคิดของ Machine Learning")
    st.write("Machine Learning (ML) คือ แนวคิดที่ทำให้คอมพิวเตอร์สามารถเรียนรู้จากข้อมูลได้เอง 📊 โดยไม่ต้องถูกตั้งโปรแกรมทุกขั้นตอน 💡")

    st.subheader("🔍 วิธีการเรียนรู้ของ Machine Learning")
    st.subheader("📌 Supervised Learning (การเรียนรู้แบบมีผู้สอน) 🎓")
    st.write("-มี ข้อมูลตัวอย่าง (Input-Output) ที่รู้คำตอบล่วงหน้า")
    st.write("-ตัวอย่างเช่น การทำนายราคาบ้าน 🏠 หรือ การจำแนกอีเมลสแปม 📩")

    st.subheader("📌 Unsupervised Learning (การเรียนรู้แบบไม่มีผู้สอน) 🧩")
    st.write("คอมพิวเตอร์ต้องหาโครงสร้างของข้อมูลเอง โดยไม่มีคำตอบให้")
    st.write("เช่น การจัดกลุ่มลูกค้า (Customer Segmentation) 🏷️")

    st.subheader("📌 Reinforcement Learning (การเรียนรู้เชิงเสริมแรง) 🏆")
    st.write("คอมพิวเตอร์เรียนรู้ผ่าน การลองผิดลองถูก และรับรางวัลเมื่อทำถูกต้อง")
    st.write("ใช้ใน เกม AI 🎮 และ หุ่นยนต์อัตโนมัติ 🤖")
    st.image("images/Animation.gif", width=500)

with tabs[5]:
    st.title("Neural Network")

    st.subheader("🔹 แนวคิดของ Neural Network")
    st.write("Neural Network (โครงข่ายประสาทเทียม) เป็นเทคนิคของ Machine Learning ที่ได้แรงบันดาลใจจาก โครงสร้างของสมองมนุษย์ 🧠💡 โดยใช้โครงข่ายของ นิวรอน (Neurons) เชื่อมต่อกันเพื่อวิเคราะห์และเรียนรู้รูปแบบของข้อมูล 📊")

    st.subheader("🔄 การทำงานของ Neural Network")
    st.write("1️⃣ Input Layer (ชั้นนำเข้า) 📥 – รับข้อมูลเข้ามา เช่น รูปภาพ 🖼️ หรือข้อความ 📝")
    st.write("2️⃣ Hidden Layers (ชั้นซ่อนเร้น) 🔄 – ประมวลผลข้อมูลผ่านการเชื่อมโยงกันของ Neurons")
    st.write("3️⃣ Output Layer (ชั้นส่งออก) 📤 – ให้ผลลัพธ์ที่ต้องการ เช่น การจำแนกรูปภาพ 🏷️")

    st.subheader("🚀 การใช้งาน Neural Network")
    st.write("✅ การรู้จำใบหน้า 📸 (Face Recognition)")
    st.write("✅ ระบบแปลภาษา 🌍 (Google Translate)")
    st.write("✅ AI ที่สามารถสร้างภาพ 🖼️ หรือแต่งเพลง 🎵")
