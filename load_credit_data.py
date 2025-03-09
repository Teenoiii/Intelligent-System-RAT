import pandas as pd

# กำหนดชื่อคอลัมน์
columns = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
    "A11", "A12", "A13", "A14", "A15", "Class"
]

# โหลดข้อมูลจาก crx.data
df = pd.read_csv("data/crx.data", names=columns, na_values="?")

# แสดงข้อมูลตัวอย่าง
print(df.head())

# ตรวจสอบ Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# บันทึกเป็น CSV 
df.to_csv("data/credit_approval_clean.csv", index=False)

print("\n✅ Data loaded and saved as 'credit_approval_clean.csv'")
