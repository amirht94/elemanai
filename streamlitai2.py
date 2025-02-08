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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import psycopg2
import boto3
from datetime import datetime

# ---------------------- تنظیمات اولیه ----------------------
st.set_page_config(page_title="مشاور هوشمند کنکور", layout="wide")
st.title("🎓 مشاور هوشمند کنکور - هوش مصنوعی برای برنامه ریزی و تحلیل نقاط ضعف")

# ---------------------- اتصال به دیتابیس ----------------------
def get_database_connection():
    try:
        # اتصال به دیتابیس ابری PostgreSQL
        return psycopg2.connect(
            host="elemankonkur.com",
            database="student_db",
            user="Administrator",
            password="yv0hrZD4ho!c_x7Yi"
        )
    except:
        # استفاده از SQLite به عنوان جایگزین
        return sqlite3.connect('student_data.db', check_same_thread=False)

conn = get_database_connection()
c = conn.cursor()

# ایجاد جداول مورد نیاز
c.execute('''
    CREATE TABLE IF NOT EXISTS scores (
        lesson TEXT,
        topic TEXT,
        score INTEGER,
        difficulty INTEGER,
        study_time FLOAT,
        error_type TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS training_history (
        timestamp DATETIME,
        accuracy FLOAT,
        features TEXT
    )
''')
conn.commit()

# ---------------------- تولید داده‌های نمونه ----------------------
def generate_sample_data():
    data = {
        "lesson": ["ریاضی", "فیزیک", "شیمی", "ادبیات", "زبان"],
        "topic": ["تابع", "سینماتیک", "استوکیومتری", "املا", "گرامر"],
        "score": [12, 14, 18, 16, 15],
        "difficulty": [4, 5, 3, 2, 2],
        "study_time": [5.5, 4.0, 3.5, 6.0, 4.5],
        "error_type": ["مفهومی", "محاسباتی", "مفهومی", "تستی", "تستی"]
    }
    return pd.DataFrame(data)

# ---------------------- آموزش مدل پیشرفته ----------------------
def train_weakness_detector(data, threshold=15):
    # پیش‌پردازش داده‌ها
    le = LabelEncoder()
    if os.path.exists('label_encoder.pkl'):
        existing_le = joblib.load('label_encoder.pkl')
        le.classes_ = existing_le.classes_
    
    data['lesson_code'] = le.fit_transform(data['lesson'])
    joblib.dump(le, 'label_encoder.pkl')
    
    # انتخاب ویژگی‌ها
    features = data[['score', 'difficulty', 'lesson_code', 'study_time']]
    data['ضعف'] = (data['score'] < threshold).astype(int)
    
    # آموزش مدل با هیپرپارامترهای پیشرفته
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # ارزیابی مدل
    X_train, X_test, y_train, y_test = train_test_split(
        features, data['ضعف'], test_size=0.2
    )
    model.fit(X_train, y_train)
    
    # ذخیره مدل
    joblib.dump(model, 'weakness_model.pkl')
    joblib.dump(features.columns.tolist(), 'feature_names.pkl')
    
    # ذخیره تاریخچه آموزش
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = model.score(X_test, y_test)
    
    conn.execute('''
        INSERT INTO training_history VALUES (?, ?, ?)
    ''', (datetime.now(), accuracy, str(features.columns.tolist())))
    conn.commit()
    
    return model, report

# ---------------------- بهینه‌سازی برنامه مطالعاتی ----------------------
def optimize_study_plan(weakness_scores, difficulties, total_hours=20):
    def objective(x):
        main_score = -np.sum(weakness_scores * difficulties * x)
        penalty = 1000 * abs(np.sum(x) - total_hours)
        return main_score + penalty

    varbounds = np.array([[0.1, total_hours]] * len(weakness_scores))
    
    algorithm_param = {
        'max_num_iteration': 200,
        'population_size': 100,
        'mutation_probability': 0.1,
        'elit_ratio': 0.1,
        'crossover_probability': 0.5,
        'crossover_type': 'uniform',
        'parents_portion': 0.3
    }
    
    model = ga(
        function=objective,
        dimension=len(weakness_scores),
        variable_type='real',
        variable_boundaries=varbounds,
        algorithm_parameters=algorithm_param
    )
    
    model.run()
    optimized_hours = model.output_dict['variable']
    optimized_hours = optimized_hours * (total_hours / np.sum(optimized_hours))
    
    return np.round(optimized_hours, 1)

# ---------------------- رابط کاربری پیشرفته ----------------------
def main():
    # مدیریت مدل اولیه
    if not os.path.exists('label_encoder.pkl'):
        st.warning("🔧 در حال آماده‌سازی اولیه سیستم...")
        df_sample = generate_sample_data()
        train_weakness_detector(df_sample)
        st.experimental_rerun()

    # بخش ورود داده‌ها
    st.sidebar.header("📤 ورود داده‌های جدید")
    with st.sidebar.form("ورود داده‌ها"):
        lesson = st.text_input("نام درس")
        topic = st.text_input("مبحث")
        score = st.number_input("نمره آزمون", 0, 20)
        difficulty = st.slider("سختی مبحث", 1, 5)
        study_time = st.number_input("زمان مطالعه هفتگی (ساعت)", 0.0, 50.0, 5.0)
        error_type = st.selectbox("نوع خطا", ["مفهومی", "محاسباتی", "تستی"])
        threshold = st.slider("آستانه تشخیص ضعف", 0, 20, 15)
        
        if st.form_submit_button("ذخیره"):
            c.execute('''
                INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)
            ''', (lesson, topic, score, difficulty, study_time, error_type))
            conn.commit()
            st.success("✅ داده‌ها با موفقیت ذخیره شدند!")

    # بخش مدیریت مدل
    st.sidebar.header("🛠 مدیریت پیشرفته")
    if st.sidebar.button("آموزش مدل جدید"):
        df = pd.read_sql("SELECT * FROM scores", conn)
        if len(df) > 0:
            try:
                model, report = train_weakness_detector(df)
                st.sidebar.success("🎉 مدل با موفقیت آموزش داده شد!")
                st.sidebar.code(f"گزارش ارزیابی:\n{report}")
            except Exception as e:
                st.sidebar.error(f"❌ خطا در آموزش مدل: {str(e)}")
        else:
            st.sidebar.warning("⚠️ ابتدا داده وارد کنید!")

    # نمایش داده‌ها
    st.subheader("📊 داده‌های ورودی")
    df = pd.read_sql("SELECT * FROM scores", conn)
    st.dataframe(df)
    
    # تحلیل داده‌ها
    if not df.empty:
        try:
            le = joblib.load('label_encoder.pkl')
            model = joblib.load('weakness_model.pkl')
            feature_names = joblib.load('feature_names.pkl')
            
            df['lesson_code'] = le.transform(df['lesson'])
            features = df[feature_names]
            df['ضعف'] = model.predict(features)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 تحلیل پیشرفته")
                fig = px.sunburst(
                    df, path=['lesson', 'topic'], values='score',
                    title='توزیع نمرات بر اساس درس و مبحث'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📚 منابع پیشنهادی")
                weak_topics = df[df['ضعف'] == 1]
                for idx, row in weak_topics.iterrows():
                    st.markdown(f"""
                    ### {row['topic']} ({row['lesson']})
                    - **نوع خطا:** {row['error_type']}
                    - **منبع آموزشی:** [فیلم آموزشی {row['topic']}](https://example.com)
                    - **کتاب پیشنهادی:** کتاب جامع {row['lesson']} انتشارات خیلی سبز
                    """)
            
            # بهینه‌سازی
            st.subheader("🎯 برنامه‌ریزی هوشمند")
            total_hours = st.slider("⏳ کل ساعات مطالعه هفتگی:", 10, 40, 20)
            
            if st.button("🔄 تولید برنامه بهینه"):
                weakness_scores = df['ضعف'].values
                difficulties = df['difficulty'].values
                optimized_hours = optimize_study_plan(weakness_scores, difficulties, total_hours)
                
                df['زمان بهینه'] = optimized_hours
                df['اولویت'] = df['ضعف'] * df['difficulty']
                df = df.sort_values('اولویت', ascending=False)
                
                st.success("✅ برنامه بهینه‌سازی شده:")
                for idx, row in df.iterrows():
                    st.progress(row['زمان بهینه']/total_hours)
                    st.write(f"""
                    **{row['lesson']} ({row['topic']})**
                    - زمان مطالعه: {row['زمان بهینه']} ساعت
                    - سطح اولویت: {row['اولویت']}/20
                    - نوع تمرکز: {"تقویت پایه" if row['error_type'] == 'مفهومی' else "تمرین تستی"}
                    """)

        except Exception as e:
            st.error(f"❌ خطا در تحلیل داده‌ها: {str(e)}")

    # بخش مدیریت حرفه‌ای
    st.sidebar.header("☁️ امکانات ابری")
    if st.sidebar.button("آپلود مدل به فضای ابری"):
        try:
            s3 = boto3.client('s3',
                aws_access_key_id='YOUR_KEY',
                aws_secret_access_key='YOUR_SECRET')
            s3.upload_file('weakness_model.pkl', 'your-bucket', 'models/latest_model.pkl')
            st.sidebar.success("✅ مدل با موفقیت آپلود شد!")
        except Exception as e:
            st.sidebar.error(f"❌ خطا در آپلود: {str(e)}")

    # گزارشات تاریخی
    st.subheader("📜 تاریخچه آموزش مدل")
    history_df = pd.read_sql("SELECT * FROM training_history ORDER BY timestamp DESC", conn)
    st.dataframe(history_df)
    
    # خروجی داده‌ها
    st.subheader("💾 مدیریت داده‌ها")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="دانلود داده‌ها (CSV)",
        data=csv,
        file_name='student_data.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()
