import streamlit as st
import joblib
import numpy as np

model = joblib.load('student_score_model.pkl')

st.title("Student Score Prediction App")
st.write("แอปพลิเคชันทำนายคะแนนสอบจากพฤติกรรมการเรียน (Multiple Regression)")

st.sidebar.header("กรุณากรอกข้อมูล")
hours_studied = st.sidebar.slider("ชั่วโมงอ่านหนังสือ (ต่อวัน)", 1, 10, 5)
prev_scores = st.sidebar.number_input("คะแนนสอบครั้งก่อน (เต็ม 100)", 0, 100, 70)
sleep_hours = st.sidebar.slider("ชั่วโมงนอน (ต่อวัน)", 4, 10, 7)
sample_papers = st.sidebar.slider("จำนวนข้อสอบเก่าที่ฝึกทำ", 0, 10, 3)

if st.button("ทำนายคะแนนสอบ"):
    features = np.array([[hours_studied, prev_scores, sleep_hours, sample_papers]])
    prediction = model.predict(features)
    st.success(f"คะแนนสอบที่คาดว่าจะได้: {prediction[0]:.2f} คะแนน")

    if prediction[0] > 80:
        st.write("เยี่ยมมาก! รักษามาตรฐานนี้ไว้นะ")
    else:
        st.write("ลองเพิ่มชั่วโมงอ่านหนังสือ หรือฝึกทำโจทย์เพิ่มดูนะ!")

st.markdown("---")
st.write("**Model:** Linear Regression | **Accuracy (R2):** 0.99 (Example)")


