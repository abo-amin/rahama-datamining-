# 🎓 Smart Academic Advisor (المرشد الأكاديمي الذكي)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-Scikit_Learn-orange.svg)

**المرشد الأكاديمي الذكي** هو تطبيق ويب متكامل مبني باستخدام تقنيات التنقيب عن البيانات (Data Mining) وتعلّم الآلة (Machine Learning). يهدف المشروع إلى تحليل السجل الأكاديمي للطلاب وتقديم توصيات دقيقة لمساعدتهم في اختيار المسار الأكاديمي الأنسب لهم، بالإضافة إلى التنبؤ بالمعدل التراكمي (GPA) بناءً على أدائهم في المقررات المختلفة ومجالات اهتمامهم.

---

## ✨ المميزات الأساسية

- 📊 **لوحة تحكم تفاعلية (Dashboard):** استكشاف بيانات الطلاب، توزيع المعدلات التراكمية، والمسارات الأكاديمية المختلفة بصرياً.
- 🏋️ **مركز تدريب النماذج (Training Center):** إمكانية تدريب نماذج تعلّم الآلة (مثل Random Forest, Decision Tree, KNN, Linear Regression) بشكل مباشر (Live Training) ومقارنة مقاييس الدقة (Accuracy, F1-Score, R2, RMSE) مع النماذج الأساسية.
- 🔮 **التنبؤ الفردي (Individual Prediction):** إدخال بيانات طالب محدد (المقررات المجتازة، الساعات المعتمدة، مجالات الاهتمام، وغيرها) للحصول على توقع للمسار الأكاديمي المناسب (التصنيف) وتوقع المعدل التراكمي المستقبلي (الانحدار).
- 🔗 **قواعد الارتباط (Association Rules):** اكتشاف العلاقات والأنماط بين المواد التي يجتازها الطلاب والمسارات الأكاديمية التي ينتهون إليها باستخدام خوارزمية **Apriori**.
- 📋 **تقارير النماذج (Model Reports):** عرض تقارير تفصيلية لأداء الخوارزميات، مصفوفات الارتباك (Confusion Matrices)، ومقارنات الانحدار.

---

## 🛠️ التقنيات المستخدمة

- **الواجهة الخلفية (Backend):** Python, Flask
- **تعلّم الآلة وتحليل البيانات:** Scikit-Learn, Pandas, NumPy, MLxtend
- **الواجهة الأمامية (Frontend):** HTML5, Vanilla CSS, JavaScript, Chart.js
- **حفظ النماذج:** Joblib

---

## 🚀 كيفية التشغيل (How to Run)

للحصول على نسخة من المشروع وتشغيلها على جهازك المحلي، اتبع الخطوات التالية:

### 1. المتطلبات المسبقة
تأكد من تثبيت [Python](https://www.python.org/downloads/) (إصدار 3.8 أو أحدث) على جهازك.

### 2. استنساخ المستودع (Clone the Repository)
قم بفتح موجّه الأوامر (Terminal/CMD) وانسخ المستودع:
```bash
git clone https://github.com/abo-amin/rahama-datamining-.git
cd Smart_Academic_Advisor
```

### 3. تثبيت المكتبات اللازمة (Install Dependencies)
يُفضل إنشاء بيئة وهمية (Virtual Environment) أولاً، ثم تثبيت المكتبات:
```bash
pip install -r requirements.txt
```

### 4. تشغيل التطبيق (Run the Application)
بعد اكتمال التثبيت، قم بتشغيل خادم الـ Flask:
```bash
python app.py
```

### 5. فتح التطبيق
افتح متصفح الويب الخاص بك وانتقل إلى الرابط التالي:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📂 هيكل المشروع (Project Structure)

```text
Smart_Academic_Advisor/
│
├── app.py                  # الملف الرئيسي لتشغيل تطبيق Flask و الـ APIs
├── trainer.py              # يحتوي على منطق تدريب النماذج وبناء الميزات
├── requirements.txt        # المكتبات والاعتماديات المطلوبة للمشروع
│
├── data/                   # يحتوي على مجموعة البيانات (CSV) والرسومات البيانية
├── models/                 # النماذج المدربة مسبقاً (Baseline) وأدوات المعالجة
├── static/                 # ملفات الـ CSS والصور (الواجهة الأمامية)
└── templates/              # ملفات الـ HTML لجميع صفحات التطبيق
```

---

## 👨‍💻 المطور
تم تصميم وتطوير هذا المشروع بواسطة **Mohamed Amin**.
