import pandas as pd

# กำหนดชื่อคอลัมน์ 
columns = [
    "Family", "Product-Type", "Steel", "Carbon", "Hardness", "Temper_rolling", "Condition",
    "Formability", "Strength", "Non-ageing", "Surface-finish", "Enamelability",
    "Bc", "Bendability", "Delamination", "Weldability", "Coating", "Lubrication",
    "Adhesion", "Quality", "Location", "Surface-defects", "Pitted", "Passivation",
    "Ferrous", "Stainless", "Resilience", "Corrosion", "Fabrication"
]

# โหลดข้อมูล
df = pd.read_csv("data/anneal.data", names=columns, na_values="?")

# แสดงตัวอย่างข้อมูล
print(df.head())

# ตรวจสอบ Missing Values
print("\nMissing Values:\n", df.isnull().sum())

# บันทึกเป็น CSV 
df.to_csv("data/anneal_clean.csv", index=False)

print("\n✅ Data loaded and saved as 'anneal_clean.csv'")
