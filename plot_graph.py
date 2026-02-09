import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Tahoma' 
mpl.rcParams['axes.unicode_minus'] = False 

df = pd.read_csv('Student_Performance.csv')
X = df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ==========================================
# 3. วาดกราฟ (ภาษาไทย)
# ==========================================
plt.figure(figsize=(10, 6))

# วาดจุด (Scatter Plot)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='#2c3e50', s=80) # ปรับสีให้เข้มขึ้นดูโปร

# วาดเส้นตรง (Perfect Prediction Line)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='เส้นความแม่นยำ 100%')

# ใส่ชื่อแกนภาษาไทย
plt.xlabel('คะแนนสอบจริง (Actual Score)', fontsize=14)
plt.ylabel('คะแนนที่ระบบทำนาย (Predicted Score)', fontsize=14)
plt.title('เปรียบเทียบ: คะแนนสอบจริง vs คะแนนที่ทำนาย', fontsize=18, fontweight='bold')
plt.legend() # โชว์คำอธิบายเส้น
plt.grid(True, linestyle='--', alpha=0.7)

# 4. เซฟรูป
plt.tight_layout()
plt.savefig('model_performance_thai.png', dpi=300) # dpi=300 รูปจะชัดระดับ HD
print("สร้างกราฟได้แล้ว")
plt.show()