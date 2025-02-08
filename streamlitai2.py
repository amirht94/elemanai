import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBClassifier
from geneticalgorithm import geneticalgorithm as ga
import joblib
import sqlite3
import os
from sklearn.preprocessing import LabelEncoder

# ---------------------- تنظیمات اولیه ----------------------
st.set_page_config(page_title="مشاور هوشمند کنکور", layout="wide")
st.title("🎓 مشاور هوشمند کنکور - هوش مصنوعی برای برنامه ریزی و تحلیل نقاط ضعف")

# ---------------------- تولید داده‌های نمونه ----------------------
def generate_sample_data():
    data = {
        "lesson": ["ریاضی", "فیزیک", "شیمی", "ادبیات", "زبان"],
        "topic": ["تابع", "سینماتیک", "استوکیومتری", "املا", "گرامر"],
        "score": [12, 14, 18, 16, 15],
        "difficulty": [4, 5, 3, 2, 2]
    }
    return pd.DataFrame(data)

# ---------------------- آموزش مدل تشخیص نقاط ضعف ----------------------
def train_weakness_detector(data, threshold=15):
    # پیش‌پردازش داده‌ها
    le = LabelEncoder()
    
    # اگر encoder قدیمی وجود دارد، کلاس‌های آن را بارگذاری کن
    if os.path.exists('label_encoder.pkl'):
        existing_le = joblib.load('label_encoder.pkl')
        le.classes_ = existing_le.classes_
    
    data['lesson_code'] = le.fit_transform(data['lesson'])
    
    # ذخیره encoder
    joblib.dump(le, 'label_encoder.pkl')
    
    features = data[['score', 'difficulty', 'lesson_code']]
    
    # برچسب‌گذاری با آستانه قابل تنظیم
    data['ضعف'] = (data['score'] < threshold).astype(int)
    
    # آموزش مدل
    model = XGBClassifier()
    model.fit(features, data['ضعف'])
    
    # ذخیره مدل و نام ویژگی‌ها
    joblib.dump(model, 'weakness_model.pkl')
    joblib.dump(features.columns.tolist(), 'feature_names.pkl')
    return model

# ---------------------- بهینه‌سازی برنامه مطالعاتی ----------------------
def optimize_study_plan(weakness_scores, difficulties, total_hours=20):
    # تابع هدف با پنالتی
    def objective(x):
        main_score = -np.sum(weakness_scores * difficulties * x)
        penalty = 1000 * abs(np.sum(x) - total_hours)
        return main_score + penalty

    # تنظیمات الگوریتم
    varbounds = np.array([[0.1, total_hours]] * len(weakness_scores))
    
    algorithm_param = {
        'max_num_iteration': 200,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.1,
        'crossover_probability': 0.5,
        'crossover_type': 'uniform',  # اضافه کردن این پارامتر
        'parents_portion': 0.3,
        'max_iteration_without_improv': 50
    }
    
    # ایجاد مدل
    model = ga(
        function=objective,
        dimension=len(weakness_scores),
        variable_type='real',
        variable_boundaries=varbounds,
        algorithm_parameters=algorithm_param
    )
    
    model.run()
    
    # نرمال‌سازی نتایج
    optimized_hours = model.output_dict['variable']
    optimized_hours = optimized_hours * (total_hours / np.sum(optimized_hours))
    
    return np.round(optimized_hours, 1)

# ---------------------- رابط کاربری Streamlit ----------------------
def main():
    # ایجاد فایل‌های اولیه اگر وجود نداشته باشند
    if not os.path.exists('label_encoder.pkl') or not os.path.exists('weakness_model.pkl'):
        st.warning("🔧 در حال آماده‌سازی اولیه سیستم...")
        df_sample = generate_sample_data()
        train_weakness_detector(df_sample)
        st.experimental_rerun()

    # بخش آپلود داده‌ها
    st.sidebar.header("📤 ورود داده‌های جدید")
    with st.sidebar.form("ورود داده‌ها"):
        lesson = st.text_input("نام درس")
        topic = st.text_input("مبحث")
        score = st.number_input("نمره آزمون", 0, 20)
        difficulty = st.slider("سختی مبحث", 1, 5)
        threshold = st.slider("آستانه تشخیص ضعف", 0, 20, 15)
        
        if st.form_submit_button("ذخیره"):
            c.execute("INSERT INTO scores VALUES (?, ?, ?, ?)", 
                     (lesson, topic, score, difficulty))
            conn.commit()
            st.success("✅ داده‌ها با موفقیت ذخیره شدند!")

    # آموزش مدل
    st.sidebar.header("🛠 آموزش مدل")
    if st.sidebar.button("آموزش مدل جدید"):
        df = pd.read_sql("SELECT * FROM scores", conn)
        if len(df) > 0:
            try:
                train_weakness_detector(df, threshold)
                st.sidebar.success("🎉 مدل با موفقیت آموزش داده شد!")
            except Exception as e:
                st.sidebar.error(f"❌ خطا در آموزش مدل: {str(e)}")
        else:
            st.sidebar.warning("⚠️ ابتدا داده وارد کنید!")

    # نمایش داده‌های موجود
    st.subheader("📊 داده‌های وارد شده")
    df = pd.read_sql("SELECT * FROM scores", conn)
    st.dataframe(df)

    # تحلیل داده‌ها
    if not df.empty:
        try:
            # بارگذاری مدل و encoder
            le = joblib.load('label_encoder.pkl')
            model = joblib.load('weakness_model.pkl')
            feature_names = joblib.load('feature_names.pkl')
            
            # بررسی کلاس‌های جدید
            current_lessons = set(df['lesson'].unique())
            trained_lessons = set(le.classes_)
            
            if not current_lessons.issubset(trained_lessons):
                st.warning("⚠️ درس‌های جدید شناسایی شده! لطفاً مدل را مجدداً آموزش دهید.")
                return
            
            # پیش‌پردازش داده‌ها
            df['lesson_code'] = le.transform(df['lesson'])
            features = df[feature_names]
            
            # پیش‌بینی
            df['ضعف'] = model.predict(features)
            
            # نمایش گرافیکی
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 تحلیل نقاط ضعف")
                fig = px.bar(df, x='topic', y='score', color='lesson', 
                            title='نمرات آزمون بر اساس مبحث')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📚 پیشنهاد منابع مطالعاتی")
                for idx, row in df[df['ضعف'] == 1].iterrows():
                    st.markdown(f"""
                    🧩 **مبحث {row['topic']} ({row['lesson']})**  
                    📚 پیشنهاد کتاب: کتاب جامع کنکور {row['lesson']} انتشارات خیلی سبز  
                    🎥 ویدیوی آموزشی: [فیلم آموزش {row['topic']}](https://example.com)
                    """)
            
            # بهینه‌سازی برنامه مطالعاتی
            st.subheader("🎯 برنامه ریزی هوشمند هفتگی")
            total_hours = st.slider("⏰ کل ساعات مطالعه هفتگی مورد نظر:", 10, 40, 20)
            
            if st.button("🔄 تولید برنامه بهینه"):
                weakness_scores = df['ضعف'].values
                difficulties = df['difficulty'].values
                
                try:
                    optimized_hours = optimize_study_plan(weakness_scores, difficulties, total_hours)
                    optimized_hours = optimized_hours * (total_hours / np.sum(optimized_hours))
                    optimized_hours = np.round(optimized_hours, 1)
                    
                    df['زمان_بهینه'] = optimized_hours
                    df['اولویت'] = df['ضعف'] * df['difficulty']
                    df = df.sort_values('اولویت', ascending=False)
                    
                    st.success("✅ برنامه بهینه سازی شده با الگوریتم هوش مصنوعی:")
                    for idx, row in df.iterrows():
                        st.markdown(f"""
                        **{row['lesson']} ({row['topic']})**  
                        ⏳ زمان مطالعه: {row['زمان_بهینه']} ساعت  
                        🎯 سطح اولویت: {row['اولویت']}/20  
                        🔍 وضعیت: {"ضعیف - نیاز به تمرکز بیشتر" if row['ضعف'] else "قوی - مرور سریع"}
                        """)
                except Exception as e:
                    st.error(f"❌ خطا در بهینه‌سازی: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ خطا در بارگذاری مدل: {str(e)}")
            return

# ---------------------- اتصال به دیتابیس ----------------------
conn = sqlite3.connect('student_data.db', check_same_thread=False)
c = conn.cursor()

# ایجاد جدول اگر وجود نداشته باشد
c.execute('''
    CREATE TABLE IF NOT EXISTS scores (
        lesson TEXT,
        topic TEXT,
        score INTEGER,
        difficulty INTEGER
    )
''')
conn.commit()

if __name__ == "__main__":
    main()
