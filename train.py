import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import m2cgen as m2c

print("=== 1. DATA CLEANING & PREPROCESSING ===")
# 1. Datasetni yuklash (Kaggle'dan olingan fayl nomi va yo'li)
try:
    df = pd.read_csv('data/student_performance.csv')
    print("Dataset muvaffaqiyatli yuklandi.")
except FileNotFoundError:
    print("Xatolik: data/student_performance.csv fayli topilmadi!")
    exit()

# 2. Keraksiz ID ustunini olib tashlash
if 'student_id' in df.columns:
    df = df.drop('student_id', axis=1)

# 3. Missing values (bo'sh kataklar) tekshirish va to'ldirish
df['weekly_self_study_hours'] = df['weekly_self_study_hours'].fillna(df['weekly_self_study_hours'].mean())
df['attendance_percentage'] = df['attendance_percentage'].fillna(df['attendance_percentage'].mean())
df['class_participation'] = df['class_participation'].fillna(df['class_participation'].mean())

# 4. Feature Engineering: Pass/Fail ustunini yaratish (Score >= 50 bo'lsa 1, aks holda 0)
df['passed'] = np.where(df['total_score'] >= 50, 1, 0)

# Xususiyatlar (Features) va Maqsadni (Target) ajratish
X = df[['weekly_self_study_hours', 'attendance_percentage', 'class_participation']]
y = df['passed']

# Train va Test ma'lumotlariga ajratish (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\n=== 2. MODEL TRAINING & TESTING (ANALYSIS) ===")
# Uchta modelni e'lon qilish
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=15, max_depth=5, random_state=42)
}

best_score = 0
best_model_name = ""
best_model_obj = None

# Modellarni o'qitish va tahlil qilish
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} modeli aniqligi (Accuracy): {score * 100:.2f}%")
    
    if score > best_score:
        best_score = score
        best_model_name = name
        best_model_obj = model

print(f"\nEng yaxshi natija: {best_model_name} ({best_score*100:.2f}%)")

print("\n=== 3. EXPORTING MODEL TO INTERFACE ===")
# Modelni JavaScript kodiga o'tkazish
raw_js_code = m2c.export_to_javascript(best_model_obj)

# Agar m2cgen funksiyani boshqacha nomlagan bo'lsa, uni standart 'predict' holatiga keltiramiz
# Kod ichidagi har qanday 'function <nom>' qismini 'function predict' ga almashtiramiz
import re
fixed_js_code = re.sub(r'function \w+\s*\(', 'function predict(', raw_js_code)

with open('model.js', 'w') as f:
    f.write("// Avtomatik generatsiya qilingan standart ML modeli\n")
    f.write(fixed_js_code)

print("Model muvaffaqiyatli 'model.js' fayliga saqlandi!")