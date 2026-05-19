import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, json, time, threading
from datetime import datetime
warnings.filterwarnings('ignore')

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score,
                              mean_squared_error, mean_absolute_error, r2_score)
import joblib

ALL_COURSES = [
    'CS101','CS102','CS201','CS202','CS210','CS220','CS301','CS302','CS310','CS320',
    'CS330','CS340','CS350','CS360','CS401','CS410','CS420','CS430','CS440','CS450',
    'MATH101','MATH102','MATH201','MATH202','MATH301','MATH302','MATH310','MATH320',
    'ENG101','ENG102','ENG201','ENG210','ENG220','ENG301','ENG310','ENG320'
]

INTEREST_MAPPING = {
    'machine learning':0,'deep learning':0,'neural networks':0,'computer vision':0,'NLP':0,'robotics':0,
    'web development':1,'mobile apps':1,'software design':1,'agile':1,'testing':1,'DevOps':1,
    'data analysis':2,'statistics':2,'visualization':2,'big data':2,'business intelligence':2,'ML':2,
    'networking':3,'routing':3,'protocols':3,'telecommunications':3,'IoT':3,'wireless':3,
    'security':4,'cryptography':4,'ethical hacking':4,'penetration testing':4,'forensics':4,'firewalls':4,
    'cloud':5,'AWS':5,'Azure':5,'virtualization':5,'containers':5,'microservices':5
}

TRACK_NAMES = ['AI','Cloud_Computing','Cybersecurity','Data_Science','Networks','Software_Engineering']
STATE_FILE = 'training_state.json'

_state_lock = threading.Lock()
training_state = {
    'is_training': False,
    'messages': [],
    'live_metrics': None,
    'started_at': None,
    'finished_at': None,
}

def _push(msg: str, cb=None):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    with _state_lock:
        training_state['messages'].append(line)
    if cb:
        cb(line)
    print(line)

def get_state_snapshot():
    with _state_lock:
        return dict(training_state)

def _save_state():
    try:
        snap = get_state_snapshot()
        snap_copy = {k: v for k, v in snap.items() if k != 'messages'}
        snap_copy['message_count'] = len(snap.get('messages', []))
        with open(STATE_FILE, 'w') as f:
            json.dump(snap_copy, f, default=str)
    except Exception:
        pass

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in ALL_COURSES:
        df[f'Passed_{c}'] = df['Courses_Passed'].apply(lambda x: 1 if c in str(x) else 0)
        df[f'Failed_{c}'] = df['Courses_Failed'].apply(lambda x: 1 if c in str(x) else 0)

    df['Total_Passed']     = df[[f'Passed_{c}' for c in ALL_COURSES]].sum(axis=1)
    df['Total_Failed']     = df[[f'Failed_{c}' for c in ALL_COURSES]].sum(axis=1)
    df['Pass_Fail_Ratio']  = df['Total_Passed'] / (df['Total_Failed'] + 1)
    df['GPA_Credit_Ratio'] = df['GPA'] / (df['Credit_Hours_Completed'] / 100.0)
    df['CS_Passed']   = df[[f'Passed_{c}' for c in ALL_COURSES if c.startswith('CS')]].sum(axis=1)
    df['Math_Passed'] = df[[f'Passed_{c}' for c in ALL_COURSES if c.startswith('MATH')]].sum(axis=1)
    df['Eng_Passed']  = df[[f'Passed_{c}' for c in ALL_COURSES if c.startswith('ENG')]].sum(axis=1)
    df['CS_Failed']   = df[[f'Failed_{c}' for c in ALL_COURSES if c.startswith('CS')]].sum(axis=1)
    df['Math_Failed'] = df[[f'Failed_{c}' for c in ALL_COURSES if c.startswith('MATH')]].sum(axis=1)
    df['Eng_Failed']  = df[[f'Failed_{c}' for c in ALL_COURSES if c.startswith('ENG')]].sum(axis=1)
    df['Interest_Encoded'] = df['Interest_Area'].astype(str).map(INTEREST_MAPPING)
    return df

def get_feature_lists(use_ie=True):
    course_features = [f'Passed_{c}' for c in ALL_COURSES] + [f'Failed_{c}' for c in ALL_COURSES]
    engineered      = ['Total_Passed','Total_Failed','Pass_Fail_Ratio','GPA_Credit_Ratio',
                       'CS_Passed','Math_Passed','Eng_Passed','CS_Failed','Math_Failed','Eng_Failed']
    numerical       = ['GPA','Credit_Hours_Completed','Attendance_Rate','Study_Hours_Per_Week','Previous_Semester_GPA']
    cat             = ['Interest_Encoded'] if use_ie else []
    feature_cols    = course_features + engineered + numerical + cat
    reg_feats       = course_features + engineered + ['Credit_Hours_Completed','Attendance_Rate',
                      'Study_Hours_Per_Week','Previous_Semester_GPA'] + cat
    return feature_cols, reg_feats

def _savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#0d1527')
    plt.close('all')

def _style_ax(ax, title=''):
    ax.set_facecolor('#0d1527')
    ax.tick_params(colors='#94a3b8')
    ax.xaxis.label.set_color('#94a3b8')
    ax.yaxis.label.set_color('#94a3b8')
    ax.title.set_color('#f1f5f9')
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')

def generate_plots(df, cls_results, reg_results, y_test_cls, y_test_reg, le, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='#0d1527')
    track_counts = df['Track'].value_counts()
    axes[0,0].bar(track_counts.index, track_counts.values, color='#6366f1', alpha=0.85)
    _style_ax(axes[0,0], 'Track Distribution')
    axes[0,0].tick_params(axis='x', rotation=30)
    axes[0,1].hist(df['GPA'], bins=30, color='#06b6d4', alpha=0.85, edgecolor='#0d1527')
    _style_ax(axes[0,1], 'GPA Distribution')
    df.boxplot(column='GPA', by='Track', ax=axes[1,0], patch_artist=True)
    _style_ax(axes[1,0], 'GPA by Track')
    axes[1,0].tick_params(axis='x', rotation=30)
    axes[1,1].scatter(df['Attendance_Rate'], df['GPA'], alpha=0.3, c='#8b5cf6', s=5)
    _style_ax(axes[1,1], 'Attendance vs GPA')
    plt.tight_layout()
    _savefig(f'{save_dir}/eda_plots.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1527')
    cls_names   = list(cls_results.keys())
    cls_acc     = [cls_results[m]['accuracy'] for m in cls_names]
    cls_f1      = [cls_results[m]['f1_weighted'] for m in cls_names]
    x = np.arange(len(cls_names)); w = 0.35
    axes[0].bar(x-w/2, cls_acc, w, label='Accuracy', color='#6366f1', alpha=0.85)
    axes[0].bar(x+w/2, cls_f1,  w, label='F1 Score', color='#06b6d4', alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cls_names, rotation=15)
    axes[0].set_ylim(0.8, 1.0); axes[0].legend()
    _style_ax(axes[0], 'Classification Models')

    reg_names   = list(reg_results.keys())
    r2_scores   = [reg_results[m]['r2'] for m in reg_names]
    rmse_scores = [reg_results[m]['rmse'] for m in reg_names]
    x2 = np.arange(len(reg_names))
    axes[1].bar(x2-w/2, r2_scores,  w, label='R2 Score', color='#10b981', alpha=0.85)
    axes[1].bar(x2+w/2, rmse_scores, w, label='RMSE',     color='#f59e0b', alpha=0.85)
    axes[1].set_xticks(x2); axes[1].set_xticklabels(reg_names, rotation=15); axes[1].legend()
    _style_ax(axes[1], 'Regression Models')
    plt.tight_layout()
    _savefig(f'{save_dir}/final_model_comparison.png')

    best_cls = max(cls_results, key=lambda k: cls_results[k]['accuracy'])
    cm = np.array(cls_results[best_cls]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='#0d1527')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=le.classes_, yticklabels=le.classes_)
    _style_ax(ax, f'Confusion Matrix — {best_cls}')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    _savefig(f'{save_dir}/confusion_matrix.png')

    best_reg = max(reg_results, key=lambda k: reg_results[k]['r2'])
    y_pred_reg = np.array(reg_results[best_reg]['y_pred'])
    y_true_reg = np.array(y_test_reg)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#0d1527')
    ax.scatter(y_true_reg, y_pred_reg, alpha=0.4, color='#06b6d4', s=8)
    mn, mx = y_true_reg.min(), y_true_reg.max()
    ax.plot([mn, mx], [mn, mx], 'r--', lw=2)
    _style_ax(ax, f'Actual vs Predicted GPA — {best_reg}')
    plt.tight_layout()
    _savefig(f'{save_dir}/regression_comparison.png')


def run_training(test_size=0.2, use_interest_encoding=True,
                 save_to_live=True, progress_callback=None) -> dict:
    t0 = time.time()
    cb = progress_callback

    _push("Loading dataset...", cb)
    df = pd.read_csv('data/academic_dataset.csv')
    _push(f"Loaded {len(df)} records", cb)

    _push("Feature engineering...", cb)
    df = build_features(df)
    feature_cols, reg_feats = get_feature_lists(use_interest_encoding)

    le = LabelEncoder()
    y_track = le.fit_transform(df['Track'].astype(str))

    _push(f"Features: {len(feature_cols)} | IE: {'yes' if use_interest_encoding else 'no'}", cb)

    X_cls = df[feature_cols]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_cls, y_track, test_size=test_size, random_state=42, stratify=y_track)
    scaler_c = StandardScaler()
    Xt_c = scaler_c.fit_transform(X_train_c)
    Xv_c = scaler_c.transform(X_test_c)

    X_reg = df[reg_feats]
    y_gpa = df['GPA']
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_gpa, test_size=test_size, random_state=42)
    scaler_r = StandardScaler()
    Xt_r = scaler_r.fit_transform(X_train_r)
    Xv_r = scaler_r.transform(X_test_r)

    _push(f"Split: train={len(X_train_c)}, test={len(X_test_c)}", cb)

    cls_defs = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=20,
                          min_samples_split=5, min_samples_leaf=2, criterion='gini'),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=200,
                          max_depth=20, min_samples_split=5, min_samples_leaf=2),
        'KNN':           KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
    }
    cls_results = {}
    for name, model in cls_defs.items():
        _push(f"Training {name} (classification)...", cb)
        model.fit(Xt_c, y_train_c)
        y_pred = model.predict(Xv_c)
        acc  = accuracy_score(y_test_c, y_pred)
        f1   = f1_score(y_test_c, y_pred, average='weighted')
        rep  = classification_report(y_test_c, y_pred, target_names=le.classes_, output_dict=True)
        cm   = confusion_matrix(y_test_c, y_pred).tolist()
        cv   = cross_val_score(model, Xt_c, y_train_c, cv=3, scoring='accuracy')
        cls_results[name] = {
            'accuracy':            round(float(acc), 4),
            'f1_weighted':         round(float(f1), 4),
            'precision_weighted':  round(float(rep['weighted avg']['precision']), 4),
            'recall_weighted':     round(float(rep['weighted avg']['recall']), 4),
            'cv_mean':             round(float(cv.mean()), 4),
            'cv_std':              round(float(cv.std()), 4),
            'per_class': {cls: {k: round(v, 4) for k, v in vals.items()}
                          for cls, vals in rep.items()
                          if cls not in ('accuracy', 'macro avg', 'weighted avg')
                          and isinstance(vals, dict)},
            'confusion_matrix': cm,
            'y_pred':    y_pred.tolist(),
            'model_obj': model,
        }
        _push(f"  {name}: Accuracy={acc:.2%}  F1={f1:.4f}  CV={cv.mean():.2%}", cb)

    reg_defs = {
        'Linear Regression':       LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=200,
                                    max_depth=20, min_samples_split=5, min_samples_leaf=2),
    }
    reg_results = {}
    for name, model in reg_defs.items():
        _push(f"Training {name} (regression)...", cb)
        model.fit(Xt_r, y_train_r)
        y_pred_r = model.predict(Xv_r)
        mse  = mean_squared_error(y_test_r, y_pred_r)
        rmse = float(np.sqrt(mse))
        mae  = mean_absolute_error(y_test_r, y_pred_r)
        r2   = r2_score(y_test_r, y_pred_r)
        cv_r = cross_val_score(model, Xt_r, y_train_r, cv=3, scoring='r2')
        reg_results[name] = {
            'mse':        round(float(mse), 4),
            'rmse':       round(rmse, 4),
            'mae':        round(float(mae), 4),
            'r2':         round(float(r2), 4),
            'cv_r2_mean': round(float(cv_r.mean()), 4),
            'cv_r2_std':  round(float(cv_r.std()), 4),
            'y_pred':    y_pred_r.tolist(),
            'model_obj': model,
        }
        _push(f"  {name}: R2={r2:.2%}  RMSE={rmse:.4f}  MAE={mae:.4f}", cb)

    best_cls = max(cls_results, key=lambda k: cls_results[k]['accuracy'])
    best_reg = max(reg_results, key=lambda k: reg_results[k]['r2'])

    elapsed = round(time.time() - t0, 1)

    save_dir  = 'models/live' if save_to_live else None
    plots_dir = 'data/live'   if save_to_live else None

    # Build metrics dict FIRST — before any disk I/O
    def _clean(r, exclude):
        return {k: v for k, v in r.items() if k not in exclude}

    metrics = {
        'classification': {n: _clean(r, ('model_obj', 'y_pred')) for n, r in cls_results.items()},
        'regression':     {n: _clean(r, ('model_obj', 'y_pred')) for n, r in reg_results.items()},
        'best_cls': best_cls,
        'best_reg': best_reg,
        'config': {
            'test_size':             test_size,
            'train_size':            len(X_train_c),
            'test_size_actual':      len(X_test_c),
            'use_interest_encoding': use_interest_encoding,
            'trained_at':            datetime.now().isoformat(),
            'elapsed_seconds':       elapsed,
        },
        'plots_dir': plots_dir or '',
        'saved_to_disk': False,
    }
    for n in cls_results:
        metrics['classification'][n]['is_best'] = (n == best_cls)
    for n in reg_results:
        metrics['regression'][n]['is_best'] = (n == best_reg)

    # Save models
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for name, res in cls_results.items():
            joblib.dump(res['model_obj'], f"{save_dir}/{name.lower().replace(' ','_')}_model.pkl")
        for name, res in reg_results.items():
            joblib.dump(res['model_obj'], f"{save_dir}/{name.lower().replace(' ','_')}_model.pkl")
        artifacts = {
            'scaler_cls': scaler_c, 'scaler_reg': scaler_r, 'label_encoder': le,
            'feature_columns': feature_cols, 'regression_features': reg_feats,
            'all_courses': ALL_COURSES, 'interest_mapping': INTEREST_MAPPING,
            'best_cls_model_name': best_cls, 'best_reg_model_name': best_reg,
            'use_interest_encoding': use_interest_encoding,
        }
        joblib.dump(artifacts, f"{save_dir}/preprocessing_artifacts.pkl")
        metrics['saved_to_disk'] = True
        _push("Models saved to models/live/", cb)

    # Generate plots — non-fatal if disk is full
    if plots_dir:
        try:
            _push("Generating plots...", cb)
            generate_plots(df, cls_results, reg_results,
                           y_test_c.tolist(), y_test_r.tolist(), le, plots_dir)
            _push("Plots saved", cb)
        except OSError as e:
            _push(f"[Warning] Could not save plots (disk full?): {e}", cb)

    metrics['_y_test_cls'] = y_test_c.tolist()
    metrics['_y_test_reg'] = y_test_r.tolist()

    _push(f"Done in {elapsed}s | Best classifier: {best_cls} ({cls_results[best_cls]['accuracy']:.2%})", cb)
    return metrics


def compute_baseline_metrics() -> dict:
    try:
        art = joblib.load('models/preprocessing_artifacts.pkl')
        df  = pd.read_csv('data/academic_dataset.csv')
        df  = build_features(df)
        le          = art['label_encoder']
        feature_cols = art['feature_columns']
        reg_feats   = art['regression_features']
        scaler_c    = art['scaler_cls']
        scaler_r    = art['scaler_reg']
        use_ie      = 'Interest_Encoded' in feature_cols

        y_track = le.transform(df['Track'].astype(str))
        X_cls   = df[feature_cols]
        _, X_test_c, _, y_test_c = train_test_split(
            X_cls, y_track, test_size=0.2, random_state=42, stratify=y_track)
        Xv_c = scaler_c.transform(X_test_c)

        X_reg  = df[reg_feats]
        y_gpa  = df['GPA']
        _, X_test_r, _, y_test_r = train_test_split(X_reg, y_gpa, test_size=0.2, random_state=42)
        Xv_r   = scaler_r.transform(X_test_r)

        cls_results = {}
        for name, fname in [('Decision Tree','decision_tree'),
                             ('Random Forest','random_forest'),
                             ('KNN','knn')]:
            m      = joblib.load(f'models/{fname}_model.pkl')
            y_pred = m.predict(Xv_c)
            acc    = accuracy_score(y_test_c, y_pred)
            f1     = f1_score(y_test_c, y_pred, average='weighted')
            rep    = classification_report(y_test_c, y_pred, target_names=le.classes_, output_dict=True)
            cm     = confusion_matrix(y_test_c, y_pred).tolist()
            cls_results[name] = {
                'accuracy':           round(float(acc), 4),
                'f1_weighted':        round(float(f1), 4),
                'precision_weighted': round(float(rep['weighted avg']['precision']), 4),
                'recall_weighted':    round(float(rep['weighted avg']['recall']), 4),
                'cv_mean': 0, 'cv_std': 0,
                'per_class': {cls: {k: round(v, 4) for k, v in vals.items()}
                              for cls, vals in rep.items()
                              if cls not in ('accuracy','macro avg','weighted avg')
                              and isinstance(vals, dict)},
                'confusion_matrix': cm,
                'is_best': False,
            }

        reg_results = {}
        for name, fname in [('Linear Regression','linear_regression'),
                             ('Random Forest Regressor','random_forest_regressor')]:
            m        = joblib.load(f'models/{fname}_model.pkl')
            y_pred_r = m.predict(Xv_r)
            mse      = mean_squared_error(y_test_r, y_pred_r)
            r2       = r2_score(y_test_r, y_pred_r)
            reg_results[name] = {
                'mse':        round(float(mse), 4),
                'rmse':       round(float(np.sqrt(mse)), 4),
                'mae':        round(float(mean_absolute_error(y_test_r, y_pred_r)), 4),
                'r2':         round(float(r2), 4),
                'cv_r2_mean': 0, 'cv_r2_std': 0,
                'is_best':    False,
            }

        best_cls = max(cls_results, key=lambda k: cls_results[k]['accuracy'])
        best_reg = max(reg_results, key=lambda k: reg_results[k]['r2'])
        cls_results[best_cls]['is_best'] = True
        reg_results[best_reg]['is_best'] = True

        return {
            'classification': cls_results,
            'regression':     reg_results,
            'best_cls': best_cls,
            'best_reg': best_reg,
            'config': {'use_interest_encoding': use_ie, 'source': 'baseline'},
        }
    except Exception as e:
        return {'error': str(e)}


def get_dataset_stats() -> dict:
    try:
        df = pd.read_csv('data/academic_dataset.csv')
        return {
            'total':    len(df),
            'tracks':   df['Track'].value_counts().to_dict(),
            'gpa_mean': round(float(df['GPA'].mean()), 2),
            'gpa_std':  round(float(df['GPA'].std()),  2),
            'gpa_min':  round(float(df['GPA'].min()),  2),
            'gpa_max':  round(float(df['GPA'].max()),  2),
            'gpa_hist': np.histogram(df['GPA'], bins=20)[0].tolist(),
            'gpa_bins': np.histogram(df['GPA'], bins=20)[1].tolist(),
        }
    except Exception as e:
        return {'error': str(e)}
