from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import threading, json, time, os, queue
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from trainer import (run_training, compute_baseline_metrics, get_dataset_stats,
                     training_state, _state_lock, _push, ALL_COURSES, INTEREST_MAPPING)

app = Flask(__name__)

# Globals
_baseline_metrics = None
_training_thread  = None

def _get_baseline():
    global _baseline_metrics
    if _baseline_metrics is None:
        _baseline_metrics = compute_baseline_metrics()
    return _baseline_metrics

def _get_live_metrics():
    with _state_lock:
        return training_state.get('live_metrics')

def _get_active_metrics():
    """Live if available, else baseline."""
    live = _get_live_metrics()
    return live if live else _get_baseline()

def _live_artifacts():
    p = 'models/live/preprocessing_artifacts.pkl'
    if os.path.exists(p):
        try:
            return joblib.load(p)
        except Exception:
            pass
    return joblib.load('models/preprocessing_artifacts.pkl')

def _build_student_input(data, feature_cols, reg_feats, all_courses):
    passed = data.get('passed_courses', [])
    failed = data.get('failed_courses', [])
    gpa    = float(data['gpa'])
    credit = int(data['credit_hours'])

    def _fill(cols):
        inp = {c: 0 for c in cols}
        for c in passed:
            if f'Passed_{c}' in inp: inp[f'Passed_{c}'] = 1
        for c in failed:
            if f'Failed_{c}' in inp: inp[f'Failed_{c}'] = 1
        inp['Total_Passed']    = len(passed)
        inp['Total_Failed']    = len(failed)
        inp['Pass_Fail_Ratio'] = len(passed) / (len(failed) + 1)
        if 'GPA_Credit_Ratio' in inp:
            inp['GPA_Credit_Ratio'] = gpa / max(credit / 100, 0.01)
        inp['CS_Passed']   = sum(1 for c in passed if c.startswith('CS'))
        inp['Math_Passed'] = sum(1 for c in passed if c.startswith('MATH'))
        inp['Eng_Passed']  = sum(1 for c in passed if c.startswith('ENG'))
        inp['CS_Failed']   = sum(1 for c in failed if c.startswith('CS'))
        inp['Math_Failed'] = sum(1 for c in failed if c.startswith('MATH'))
        inp['Eng_Failed']  = sum(1 for c in failed if c.startswith('ENG'))
        if 'GPA' in inp:             inp['GPA']                    = gpa
        if 'Credit_Hours_Completed' in inp: inp['Credit_Hours_Completed'] = credit
        inp['Attendance_Rate']       = float(data.get('attendance', 80))
        inp['Study_Hours_Per_Week']  = float(data.get('study_hours', 15))
        inp['Previous_Semester_GPA'] = float(data.get('prev_gpa', gpa))
        if 'Interest_Encoded' in inp:
            inp['Interest_Encoded'] = INTEREST_MAPPING.get(data.get('interest', ''), 0)
        return inp

    return _fill(feature_cols), _fill(reg_feats)

@app.route('/')
def dashboard():
    stats = get_dataset_stats()
    return render_template('dashboard.html', stats=stats, courses=ALL_COURSES, active='dashboard')

@app.route('/training')
def training():
    return render_template('training.html', courses=ALL_COURSES, active='training')

@app.route('/reports')
def reports():
    return render_template('reports.html', active='reports')

@app.route('/predict')
def predict_page():
    interest_areas = list(INTEREST_MAPPING.keys())
    return render_template('predict.html', courses=ALL_COURSES, interest_areas=interest_areas, active='predict')

@app.route('/tests')
def tests_page():
    return render_template('tests.html', active='tests')

@app.route('/association')
def association_page():
    return render_template('association.html', active='association')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/api/metrics')
def api_metrics():
    baseline = _get_baseline()
    live     = _get_live_metrics()
    stats    = get_dataset_stats()
    # Add plot timestamps to bust cache
    def plot_ts(path):
        try:
            return int(os.path.getmtime(path) * 1000)
        except:
            return 0

    live_plots = {
        'eda':        f'/data/live/eda_plots.png?v={plot_ts("data/live/eda_plots.png")}',
        'final':      f'/data/live/final_model_comparison.png?v={plot_ts("data/live/final_model_comparison.png")}',
        'regression': f'/data/live/regression_comparison.png?v={plot_ts("data/live/regression_comparison.png")}',
        'confusion':  f'/data/live/confusion_matrix.png?v={plot_ts("data/live/confusion_matrix.png")}',
    } if live else {}
    base_plots = {
        'eda':        f'/data/eda_plots.png?v={plot_ts("data/eda_plots.png")}',
        'final':      f'/data/final_model_comparison.png?v={plot_ts("data/final_model_comparison.png")}',
        'regression': f'/data/regression_comparison.png?v={plot_ts("data/regression_comparison.png")}',
    }
    return jsonify({
        'baseline': baseline,
        'live':     live,
        'dataset':  stats,
        'plots': {'live': live_plots, 'baseline': base_plots},
    })

@app.route('/api/training-status')
def api_training_status():
    since = int(request.args.get('since', 0))
    with _state_lock:
        is_training = training_state['is_training']
        messages    = training_state['messages'][since:]
        total       = len(training_state['messages'])
        live        = training_state.get('live_metrics')
    return jsonify({
        'is_training':             is_training,
        'messages':                messages,
        'total_messages':          total,
        'has_live':                live is not None,
    })

@app.route('/api/association-rules')
def api_association_rules():
    try:
        art = _live_artifacts()
        if 'association_rules' in art:
            rules_dict = art['association_rules']
            df_rules = pd.DataFrame(rules_dict)
            formatted = []
            for _, row in df_rules.iterrows():
                formatted.append({
                    'antecedents': list(row['antecedents']),
                    'consequents': list(row['consequents']),
                    'support': round(float(row['support']), 4),
                    'confidence': round(float(row['confidence']), 4),
                    'lift': round(float(row['lift']), 4)
                })
            return jsonify({'rules': formatted})
    except Exception as e:
        pass

    try:
        df = pd.read_csv('data/academic_dataset.csv')
        for course in ALL_COURSES:
            df[f'Passed_{course}'] = df['Courses_Passed'].apply(lambda x: 1 if course in str(x) else 0)
        
        df['Interest_Encoded'] = df['Interest_Area'].astype(str).map(INTEREST_MAPPING)

        transactions = []
        for idx, row in df.iterrows():
            t = []
            for course in ALL_COURSES:
                if row.get(f'Passed_{course}', 0) == 1:
                    t.append(course)
            t.append(f'Interest_{row.get("Interest_Encoded", "")}')
            if row['GPA'] >= 3.5:
                t.append('High_GPA')
            elif row['GPA'] >= 2.5:
                t.append('Medium_GPA')
            else:
                t.append('Low_GPA')
            t.append(f'Track_{row["Track"]}')
            transactions.append(t)
        
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_transactions = pd.DataFrame(te_ary, columns=te.columns_)
        
        # Lower min_support and max_len for faster dynamic calculation
        frequent_itemsets = apriori(df_transactions, min_support=0.12, use_colnames=True, max_len=3)
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
        
        track_rules = rules[rules['consequents'].apply(lambda x: any('Track_' in str(item) for item in x))]
        track_rules = track_rules.sort_values('confidence', ascending=False)
        
        formatted = []
        for _, row in track_rules.head(50).iterrows():
            formatted.append({
                'antecedents': list(row['antecedents']),
                'consequents': list(row['consequents']),
                'support': round(float(row['support']), 4),
                'confidence': round(float(row['confidence']), 4),
                'lift': round(float(row['lift']), 4)
            })
        return jsonify({'rules': formatted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def api_train():
    global _training_thread
    with _state_lock:
        if training_state['is_training']:
            return jsonify({'error': 'تدريب جارٍ بالفعل'}), 409

    data       = request.json or {}
    test_size  = float(data.get('test_size', 0.2))
    use_ie     = bool(data.get('use_interest_encoding', True))

    def _do_train():
        with _state_lock:
            training_state['is_training'] = True
            training_state['messages'] = []
            training_state['started_at'] = datetime.now().isoformat()

        try:
            metrics = run_training(
                test_size=test_size,
                use_interest_encoding=use_ie,
                save_to_live=True,
                progress_callback=None,
            )
            # strip large lists before storing
            for cls_res in metrics.get('classification', {}).values():
                cls_res.pop('y_pred', None)
            for reg_res in metrics.get('regression', {}).values():
                reg_res.pop('y_pred', None)
            metrics.pop('_y_test_cls', None)
            metrics.pop('_y_test_reg', None)

            with _state_lock:
                training_state['live_metrics'] = metrics

        except Exception as e:
            _push(f"❌ خطأ: {e}")
        finally:
            with _state_lock:
                training_state['is_training'] = False
                training_state['finished_at'] = datetime.now().isoformat()

    _training_thread = threading.Thread(target=_do_train, daemon=True)
    _training_thread.start()
    return jsonify({'status': 'started'})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset to baseline: delete live models."""
    import shutil
    try:
        if os.path.exists('models/live'):
            shutil.rmtree('models/live')
        if os.path.exists('data/live'):
            shutil.rmtree('data/live')
        with _state_lock:
            training_state['live_metrics'] = None
            training_state['messages'] = []
        global _baseline_metrics
        _baseline_metrics = None
        return jsonify({'status': 'reset_done'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        art  = _live_artifacts()
        le   = art['label_encoder']
        fc   = art['feature_columns']
        rf   = art['regression_features']
        sc_c = art['scaler_cls']
        sc_r = art['scaler_reg']
        best_c = art['best_cls_model_name']
        best_r = art['best_reg_model_name']

        live_model_path = f"models/live/{best_c.lower().replace(' ','_')}_model.pkl"
        is_live = os.path.exists(live_model_path)
        base_dir = 'models/live' if is_live else 'models'
        cls_model = joblib.load(f"{base_dir}/{best_c.lower().replace(' ','_')}_model.pkl")
        reg_model = joblib.load(f"{base_dir}/{best_r.lower().replace(' ','_')}_model.pkl")

        inp_cls, inp_reg = _build_student_input(data, fc, rf, ALL_COURSES)

        X_cls = pd.DataFrame([inp_cls])
        X_cls_s = sc_c.transform(X_cls)
        pred  = cls_model.predict(X_cls_s)[0]
        probs = cls_model.predict_proba(X_cls_s)[0]
        track = le.inverse_transform([pred])[0]
        conf  = float(probs[pred])
        top3  = sorted(zip(le.classes_, probs.tolist()), key=lambda x: x[1], reverse=True)[:3]

        X_reg = pd.DataFrame([inp_reg])
        X_reg_s = sc_r.transform(X_reg)
        gpa_pred = float(np.clip(reg_model.predict(X_reg_s)[0], 1.5, 4.0))

        return jsonify({
            'track':       track,
            'confidence':  round(conf * 100, 1),
            'top3':        [{'track': t, 'prob': round(float(p)*100,1)} for t,p in top3],
            'predicted_gpa': round(gpa_pred, 2),
            'model_source': 'live' if is_live else 'baseline'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

TEST_CASES = [
    {
        'name': 'طالب AI متفوق',
        'expected_track': 'AI',
        'data': {
            'gpa': 3.8, 'credit_hours': 110, 'attendance': 92, 'study_hours': 22, 'prev_gpa': 3.75,
            'interest': 'deep learning',
            'passed_courses': ['CS101','CS201','CS301','CS302','CS401','CS410','MATH101','MATH201','MATH301','MATH302'],
            'failed_courses': [],
        }
    },
    {
        'name': 'طالب Cybersecurity',
        'expected_track': 'Cybersecurity',
        'data': {
            'gpa': 3.2, 'credit_hours': 95, 'attendance': 85, 'study_hours': 18, 'prev_gpa': 3.1,
            'interest': 'security',
            'passed_courses': ['CS330','CS340','CS350','CS360','ENG101','ENG210'],
            'failed_courses': ['CS101'],
        }
    },
    {
        'name': 'طالب Cloud Computing',
        'expected_track': 'Cloud_Computing',
        'data': {
            'gpa': 3.5, 'credit_hours': 100, 'attendance': 88, 'study_hours': 20, 'prev_gpa': 3.4,
            'interest': 'cloud',
            'passed_courses': ['CS420','CS430','CS440','CS450','ENG210','ENG310'],
            'failed_courses': [],
        }
    },
    {
        'name': 'طالب Software Engineering',
        'expected_track': 'Software_Engineering',
        'data': {
            'gpa': 2.9, 'credit_hours': 120, 'attendance': 75, 'study_hours': 14, 'prev_gpa': 2.8,
            'interest': 'web development',
            'passed_courses': ['CS210','CS220','CS310','CS320','ENG210','ENG310','ENG320'],
            'failed_courses': [],
        }
    },
    {
        'name': 'طالب Data Science',
        'expected_track': 'Data_Science',
        'data': {
            'gpa': 3.6, 'credit_hours': 105, 'attendance': 90, 'study_hours': 19, 'prev_gpa': 3.5,
            'interest': 'data analysis',
            'passed_courses': ['CS301','CS401','CS440','MATH201','MATH301','MATH302','ENG220'],
            'failed_courses': [],
        }
    },
    {
        'name': 'طالب Networks',
        'expected_track': 'Networks',
        'data': {
            'gpa': 3.1, 'credit_hours': 90, 'attendance': 80, 'study_hours': 16, 'prev_gpa': 3.0,
            'interest': 'networking',
            'passed_courses': ['CS202','CS210','CS310','CS320','ENG210'],
            'failed_courses': [],
        }
    },
]

@app.route('/api/run-tests', methods=['POST'])
def api_run_tests():
    results = []
    try:
        art    = _live_artifacts()
        le     = art['label_encoder']
        fc     = art['feature_columns']
        rf     = art['regression_features']
        sc_c   = art['scaler_cls']
        sc_r   = art['scaler_reg']
        best_c = art['best_cls_model_name']
        live_model_path = f"models/live/{best_c.lower().replace(' ','_')}_model.pkl"
        is_live = os.path.exists(live_model_path)
        base_dir = 'models/live' if is_live else 'models'
        cls_model = joblib.load(f"{base_dir}/{best_c.lower().replace(' ','_')}_model.pkl")

        for tc in TEST_CASES:
            inp, _ = _build_student_input(tc['data'], fc, rf, ALL_COURSES)
            X = pd.DataFrame([inp])
            Xs = sc_c.transform(X)
            pred  = cls_model.predict(Xs)[0]
            probs = cls_model.predict_proba(Xs)[0]
            track = le.inverse_transform([pred])[0]
            conf  = float(probs[pred])
            passed_test = track == tc['expected_track']
            results.append({
                'name':     tc['name'],
                'expected': tc['expected_track'],
                'got':      track,
                'confidence': round(conf*100, 1),
                'passed':   passed_test,
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    total   = len(results)
    passing = sum(1 for r in results if r['passed'])
    return jsonify({
        'results': results, 
        'total': total, 
        'passed': passing, 
        'failed': total - passing,
        'model_source': 'live' if is_live else 'baseline'
    })

@app.route('/api/test-cases')
def api_test_cases():
    return jsonify(TEST_CASES)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
