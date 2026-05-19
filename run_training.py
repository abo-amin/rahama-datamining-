import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib
import os

print('='*60)
print('SMART ACADEMIC ADVISOR - DATA MINING & ML')
print('='*60)

print('\n[1/10] Loading dataset...')
df = pd.read_csv('data/academic_dataset.csv')
print(f'Dataset Shape: {df.shape}')

print('\n[2/10] Feature Engineering...')
all_courses = ['CS101', 'CS102', 'CS201', 'CS202', 'CS210', 'CS220', 'CS301', 'CS302', 'CS310', 'CS320',
               'CS330', 'CS340', 'CS350', 'CS360', 'CS401', 'CS410', 'CS420', 'CS430', 'CS440', 'CS450',
               'MATH101', 'MATH102', 'MATH201', 'MATH202', 'MATH301', 'MATH302', 'MATH310', 'MATH320',
               'ENG101', 'ENG102', 'ENG201', 'ENG210', 'ENG220', 'ENG301', 'ENG310', 'ENG320']

for course in all_courses:
    df[f'Passed_{course}'] = df['Courses_Passed'].apply(lambda x: 1 if course in str(x) else 0)
    df[f'Failed_{course}'] = df['Courses_Failed'].apply(lambda x: 1 if course in str(x) else 0)

df['Total_Passed'] = df[[f'Passed_{c}' for c in all_courses]].sum(axis=1)
df['Total_Failed'] = df[[f'Failed_{c}' for c in all_courses]].sum(axis=1)
df['Pass_Fail_Ratio'] = df['Total_Passed'] / (df['Total_Failed'] + 1)
df['GPA_Credit_Ratio'] = df['GPA'] / (df['Credit_Hours_Completed'] / 100)
df['CS_Passed'] = df[[f'Passed_{c}' for c in all_courses if c.startswith('CS')]].sum(axis=1)
df['Math_Passed'] = df[[f'Passed_{c}' for c in all_courses if c.startswith('MATH')]].sum(axis=1)
df['Eng_Passed'] = df[[f'Passed_{c}' for c in all_courses if c.startswith('ENG')]].sum(axis=1)
df['CS_Failed'] = df[[f'Failed_{c}' for c in all_courses if c.startswith('CS')]].sum(axis=1)
df['Math_Failed'] = df[[f'Failed_{c}' for c in all_courses if c.startswith('MATH')]].sum(axis=1)
df['Eng_Failed'] = df[[f'Failed_{c}' for c in all_courses if c.startswith('ENG')]].sum(axis=1)

interest_mapping = {
    'machine learning': 0, 'deep learning': 0, 'neural networks': 0, 'computer vision': 0, 'NLP': 0, 'robotics': 0,
    'web development': 1, 'mobile apps': 1, 'software design': 1, 'agile': 1, 'testing': 1, 'DevOps': 1,
    'data analysis': 2, 'statistics': 2, 'visualization': 2, 'big data': 2, 'business intelligence': 2, 'ML': 2,
    'networking': 3, 'routing': 3, 'protocols': 3, 'telecommunications': 3, 'IoT': 3, 'wireless': 3,
    'security': 4, 'cryptography': 4, 'ethical hacking': 4, 'penetration testing': 4, 'forensics': 4, 'firewalls': 4,
    'cloud': 5, 'AWS': 5, 'Azure': 5, 'virtualization': 5, 'containers': 5, 'microservices': 5
}
df['Interest_Encoded'] = df['Interest_Area'].map(interest_mapping)
print(f'Created {len(all_courses)*2 + 10 + 1} features')

print('\n[3/10] Association Rules Mining (Apriori)...')
transactions = []
for idx, row in df.iterrows():
    t = []
    for course in all_courses:
        if row[f'Passed_{course}'] == 1:
            t.append(course)
    t.append(f'Interest_{row["Interest_Encoded"]}')
    if row['GPA'] >= 3.5:
        t.append('High_GPA')
    elif row['GPA'] >= 2.5:
        t.append('Medium_GPA')
    else:
        t.append('Low_GPA')
    t.append(f'Track_{row["Track"]}')
    transactions.append(t)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_transactions = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_transactions, min_support=0.1, use_colnames=True, max_len=4)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules = rules.sort_values('confidence', ascending=False)

track_rules = rules[rules['consequents'].apply(lambda x: any('Track_' in str(item) for item in x))]
track_rules = track_rules.sort_values('confidence', ascending=False)

print(f'Found {len(frequent_itemsets)} frequent itemsets')
print(f'Found {len(rules)} association rules')
print(f'Found {len(track_rules)} track-related rules')

print('\n[4/10] Preparing Classification data (Track Prediction)...')
course_features = [f'Passed_{c}' for c in all_courses] + [f'Failed_{c}' for c in all_courses]
engineered_features = ['Total_Passed', 'Total_Failed', 'Pass_Fail_Ratio', 'GPA_Credit_Ratio',
                       'CS_Passed', 'Math_Passed', 'Eng_Passed', 'CS_Failed', 'Math_Failed', 'Eng_Failed']
numerical_features = ['GPA', 'Credit_Hours_Completed', 'Attendance_Rate', 'Study_Hours_Per_Week', 'Previous_Semester_GPA']
categorical_features = ['Interest_Encoded']

feature_columns = course_features + engineered_features + numerical_features + categorical_features

X = df[feature_columns]
y_track = df['Track']

le = LabelEncoder()
y_track_encoded = le.fit_transform(y_track)

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_track_encoded, test_size=0.2, random_state=42, stratify=y_track_encoded)

scaler_cls = StandardScaler()
X_train_cls_scaled = scaler_cls.fit_transform(X_train_cls)
X_test_cls_scaled = scaler_cls.transform(X_test_cls)

print(f'Training: {X_train_cls.shape[0]}, Test: {X_test_cls.shape[0]}')

print('\n[5/10] Preparing Regression data (GPA Prediction)...')
regression_features = course_features + engineered_features + ['Credit_Hours_Completed', 'Attendance_Rate', 'Study_Hours_Per_Week', 'Previous_Semester_GPA', 'Interest_Encoded']

X_reg = df[regression_features]
y_gpa = df['GPA']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_gpa, test_size=0.2, random_state=42)

scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print(f'Training: {X_train_reg.shape[0]}, Test: {X_test_reg.shape[0]}')

print('\n[6/10] Training Classification models...')
cls_models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20, min_samples_split=5, min_samples_leaf=2, criterion='gini'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2),
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan')
}

cls_results = {}
for name, model in cls_models.items():
    model.fit(X_train_cls_scaled, y_train_cls)
    y_pred = model.predict(X_test_cls_scaled)
    acc = accuracy_score(y_test_cls, y_pred)
    f1 = f1_score(y_test_cls, y_pred, average='weighted')
    cls_results[name] = {'model': model, 'accuracy': acc, 'f1_score': f1, 'predictions': y_pred}
    print(f'{name}: Accuracy={acc:.4f}, F1={f1:.4f}')

print('\n[7/10] Training Regression models...')
reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2)
}

reg_results = {}
for name, model in reg_models.items():
    model.fit(X_train_reg_scaled, y_train_reg)
    y_pred_reg = model.predict(X_test_reg_scaled)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    reg_results[name] = {'model': model, 'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2, 'predictions': y_pred_reg}
    print(f'{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')

print('\n[8/10] Final Model Comparison...')
print('='*70)
print('CLASSIFICATION (Track Prediction)')
print('='*70)
for name, res in cls_results.items():
    print(f'{name}: Accuracy={res["accuracy"]:.4f}, F1={res["f1_score"]:.4f}')

best_cls = max(cls_results, key=lambda k: cls_results[k]['accuracy'])
print(f'\n*** BEST CLASSIFICATION: {best_cls} (Accuracy: {cls_results[best_cls]["accuracy"]:.4f}) ***')

print('\n' + '='*70)
print('REGRESSION (GPA Prediction)')
print('='*70)
for name, res in reg_results.items():
    print(f'{name}: R2={res["r2"]:.4f}, RMSE={res["rmse"]:.4f}, MAE={res["mae"]:.4f}')

best_reg = max(reg_results, key=lambda k: reg_results[k]['r2'])
print(f'\n*** BEST REGRESSION: {best_reg} (R2: {reg_results[best_reg]["r2"]:.4f}) ***')

print('\nClassification Report:')
print(classification_report(y_test_cls, cls_results[best_cls]['predictions'], target_names=le.classes_))

print('\n[9/10] Saving models...')
os.makedirs('models', exist_ok=True)

for name, res in cls_results.items():
    filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(res['model'], filename)
    print(f'Saved {name} (Classification)')

for name, res in reg_results.items():
    filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
    joblib.dump(res['model'], filename)
    print(f'Saved {name} (Regression)')

artifacts = {
    'scaler_cls': scaler_cls, 'scaler_reg': scaler_reg, 'label_encoder': le,
    'feature_columns': feature_columns, 'regression_features': regression_features,
    'all_courses': all_courses, 'best_cls_model_name': best_cls, 'best_reg_model_name': best_reg,
    'association_rules': track_rules.head(50).to_dict()
}
joblib.dump(artifacts, 'models/preprocessing_artifacts.pkl')
print('Saved preprocessing artifacts')

print('\n[10/10] Testing predictions...')
def predict_track(gpa, credit_hours, passed_courses, failed_courses, interest_encoded, attendance, study_hours, prev_gpa):
    input_data = {col: 0 for col in feature_columns}
    for course in passed_courses:
        if f'Passed_{course}' in input_data:
            input_data[f'Passed_{course}'] = 1
    for course in failed_courses:
        if f'Failed_{course}' in input_data:
            input_data[f'Failed_{course}'] = 1
    input_data['Total_Passed'] = len(passed_courses)
    input_data['Total_Failed'] = len(failed_courses)
    input_data['Pass_Fail_Ratio'] = len(passed_courses) / (len(failed_courses) + 1)
    input_data['GPA_Credit_Ratio'] = gpa / (credit_hours / 100)
    input_data['CS_Passed'] = sum(1 for c in passed_courses if c.startswith('CS'))
    input_data['Math_Passed'] = sum(1 for c in passed_courses if c.startswith('MATH'))
    input_data['Eng_Passed'] = sum(1 for c in passed_courses if c.startswith('ENG'))
    input_data['CS_Failed'] = sum(1 for c in failed_courses if c.startswith('CS'))
    input_data['Math_Failed'] = sum(1 for c in failed_courses if c.startswith('MATH'))
    input_data['Eng_Failed'] = sum(1 for c in failed_courses if c.startswith('ENG'))
    input_data['GPA'] = gpa
    input_data['Credit_Hours_Completed'] = credit_hours
    input_data['Attendance_Rate'] = attendance
    input_data['Study_Hours_Per_Week'] = study_hours
    input_data['Previous_Semester_GPA'] = prev_gpa
    input_data['Interest_Encoded'] = interest_encoded
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler_cls.transform(input_df)
    best_model = cls_results[best_cls]['model']
    prediction = best_model.predict(input_scaled)[0]
    probabilities = best_model.predict_proba(input_scaled)[0]
    track_name = le.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    track_probs = list(zip(le.classes_, probabilities))
    track_probs.sort(key=lambda x: x[1], reverse=True)
    return track_name, confidence, track_probs[:3]

def predict_gpa(credit_hours, passed_courses, failed_courses, interest_encoded, attendance, study_hours, prev_gpa):
    input_data = {col: 0 for col in regression_features}
    for course in passed_courses:
        if f'Passed_{course}' in input_data:
            input_data[f'Passed_{course}'] = 1
    for course in failed_courses:
        if f'Failed_{course}' in input_data:
            input_data[f'Failed_{course}'] = 1
    input_data['Total_Passed'] = len(passed_courses)
    input_data['Total_Failed'] = len(failed_courses)
    input_data['Pass_Fail_Ratio'] = len(passed_courses) / (len(failed_courses) + 1)
    input_data['GPA_Credit_Ratio'] = 3.0 / (credit_hours / 100)
    input_data['CS_Passed'] = sum(1 for c in passed_courses if c.startswith('CS'))
    input_data['Math_Passed'] = sum(1 for c in passed_courses if c.startswith('MATH'))
    input_data['Eng_Passed'] = sum(1 for c in passed_courses if c.startswith('ENG'))
    input_data['CS_Failed'] = sum(1 for c in failed_courses if c.startswith('CS'))
    input_data['Math_Failed'] = sum(1 for c in failed_courses if c.startswith('MATH'))
    input_data['Eng_Failed'] = sum(1 for c in failed_courses if c.startswith('ENG'))
    input_data['Credit_Hours_Completed'] = credit_hours
    input_data['Attendance_Rate'] = attendance
    input_data['Study_Hours_Per_Week'] = study_hours
    input_data['Previous_Semester_GPA'] = prev_gpa
    input_data['Interest_Encoded'] = interest_encoded
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler_reg.transform(input_df)
    best_reg_model = reg_results[best_reg]['model']
    predicted_gpa = best_reg_model.predict(input_scaled)[0]
    return float(np.clip(predicted_gpa, 1.5, 4.0))

test_passed = ['CS101', 'CS102', 'CS201', 'CS301', 'CS302', 'CS401', 'MATH101', 'MATH201', 'MATH301', 'MATH302', 'ENG101']
test_failed = ['CS220']

track, conf, top3 = predict_track(gpa=3.45, credit_hours=100, passed_courses=test_passed, failed_courses=test_failed, interest_encoded=0, attendance=88.5, study_hours=18.0, prev_gpa=3.50)
print(f'Track Prediction: {track} ({conf:.2%})')
for t, p in top3:
    print(f'  {t}: {p:.2%}')

predicted_gpa = predict_gpa(credit_hours=100, passed_courses=test_passed, failed_courses=test_failed, interest_encoded=0, attendance=88.5, study_hours=18.0, prev_gpa=3.50)
print(f'GPA Prediction: {predicted_gpa:.2f}')

print('\n' + '='*60)
print('COMPLETE!')
print('='*60)
