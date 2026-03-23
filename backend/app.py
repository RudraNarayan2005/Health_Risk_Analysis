"""
Predictive Health Risk Analysis — Flask Backend v5.0
✅ Local MySQL — data persists across PyCharm restarts
✅ Auto-creates tables on first run (no manual SQL needed)
✅ .env file for credentials — never hardcode passwords
✅ Graceful fallback to in-memory if MySQL not running
✅ 5 ML models auto-trained on startup
"""

import os, json
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()   # reads .env file from project root

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, '..', 'frontend')
MODEL_DIR    = os.path.join(BASE_DIR, '..', 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

app      = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
DISEASES = ['diabetes', 'heart_disease', 'hypertension', 'stroke', 'obesity']
MODELS   = {}

# ── Database state ─────────────────────────────────────────────
DB_ENABLED = False          # set to True when MySQL connects

# In-memory fallback
patients_db        = {}
assessments_db     = []
assessment_counter = [1]

# ── DB helpers ─────────────────────────────────────────────────
def get_db():
    import mysql.connector
    return mysql.connector.connect(
        host       = os.getenv('DB_HOST', 'localhost'),
        port       = int(os.getenv('DB_PORT', 3306)),
        user       = os.getenv('DB_USER', 'root'),
        password   = os.getenv('DB_PASSWORD', ''),
        database   = os.getenv('DB_NAME', 'health_risk_db'),
        charset    = 'utf8mb4',
        autocommit = True,
    )

def db_execute(query, params=None, fetch=True):
    conn = get_db()
    cur  = conn.cursor(dictionary=True)
    cur.execute(query, params or ())
    if fetch:
        result = cur.fetchall()
        cur.close(); conn.close()
        return result
    last_id = cur.lastrowid
    cur.close(); conn.close()
    return last_id

def init_db():
    global DB_ENABLED
    if not os.getenv('DB_HOST'):
        print("  [DB] DB_HOST not in .env — using in-memory storage.")
        return
    try:
        import mysql.connector
        conn = mysql.connector.connect(
            host       = os.getenv('DB_HOST', 'localhost'),
            port       = int(os.getenv('DB_PORT', 3306)),
            user       = os.getenv('DB_USER', 'root'),
            password   = os.getenv('DB_PASSWORD', ''),
            database   = os.getenv('DB_NAME', 'health_risk_db'),
            charset    = 'utf8mb4',
            autocommit = True,
        )
        cur = conn.cursor()

        # patients
        cur.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id         INT AUTO_INCREMENT PRIMARY KEY,
                patient_id VARCHAR(10)  NOT NULL UNIQUE,
                name       VARCHAR(100) NOT NULL,
                age        INT          NOT NULL,
                gender     VARCHAR(20)  DEFAULT 'Unknown',
                email      VARCHAR(150) DEFAULT '',
                phone      VARCHAR(30)  DEFAULT '',
                created_at DATETIME     DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_pid (patient_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        # health_assessments
        cur.execute("""
            CREATE TABLE IF NOT EXISTS health_assessments (
                id                          INT AUTO_INCREMENT PRIMARY KEY,
                patient_id                  VARCHAR(10)  NOT NULL,
                age                         INT,
                gender                      VARCHAR(20),
                bmi                         FLOAT,
                blood_pressure_systolic     INT,
                blood_pressure_diastolic    INT,
                blood_glucose               FLOAT,
                cholesterol                 FLOAT,
                heart_rate                  INT,
                smoking                     TINYINT(1)  DEFAULT 0,
                alcohol_consumption         VARCHAR(20) DEFAULT 'None',
                physical_activity           VARCHAR(20) DEFAULT 'Sedentary',
                diet_quality                VARCHAR(20) DEFAULT 'Fair',
                sleep_hours                 FLOAT,
                stress_level                FLOAT,
                family_history_diabetes     TINYINT(1)  DEFAULT 0,
                family_history_heart        TINYINT(1)  DEFAULT 0,
                family_history_hypertension TINYINT(1)  DEFAULT 0,
                family_history_cancer       TINYINT(1)  DEFAULT 0,
                risk_predictions            JSON,
                overall_risk_score          FLOAT,
                risk_category               VARCHAR(20),
                risk_factors                JSON,
                recommendations             JSON,
                alerts                      JSON,
                assessed_at                 DATETIME    DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_pid (patient_id),
                INDEX idx_cat (risk_category)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        cur.close(); conn.close()
        DB_ENABLED = True
        print(f"  [DB] Connected to MySQL — {os.getenv('DB_NAME')} on localhost:{os.getenv('DB_PORT',3306)}")
        print("  [DB] Tables ready (patients, health_assessments).")

    except Exception as e:
        print(f"  [DB] MySQL error: {e}")
        print("  [DB] Check: MySQL service running? Password correct? DB exists?")
        print("  [DB] Falling back to in-memory.")

# ── Patient CRUD ───────────────────────────────────────────────
def _next_pid():
    if DB_ENABLED:
        n = db_execute("SELECT COUNT(*) as c FROM patients")[0]['c'] + 1
    else:
        n = max([int(k[1:]) for k in patients_db if k[1:].isdigit()], default=0) + 1
    return f"P{str(n).zfill(3)}"

def db_create_patient(data):
    pid = _next_pid()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    p   = {'patient_id': pid, 'name': data['name'],
           'age': int(data.get('age', 0)), 'gender': data.get('gender', 'Unknown'),
           'email': data.get('email', ''), 'phone': data.get('phone', ''), 'created_at': now}
    if DB_ENABLED:
        db_execute(
            "INSERT INTO patients (patient_id,name,age,gender,email,phone) VALUES(%s,%s,%s,%s,%s,%s)",
            (pid, p['name'], p['age'], p['gender'], p['email'], p['phone']), fetch=False)
    else:
        patients_db[pid] = p
    return p

def db_get_patient(pid):
    if DB_ENABLED:
        rows = db_execute("SELECT * FROM patients WHERE patient_id=%s", (pid,))
        if rows:
            r = rows[0]; r['created_at'] = str(r.get('created_at',''))
            return r
        return None
    return patients_db.get(pid)

def db_get_all_patients():
    if DB_ENABLED:
        rows = db_execute("""
            SELECT p.*,
                   COUNT(a.id) AS assessment_count,
                   MAX(a.overall_risk_score) AS overall_risk_score,
                   MAX(a.assessed_at) AS latest_assessment_date,
                   (SELECT risk_category FROM health_assessments
                    WHERE patient_id=p.patient_id ORDER BY assessed_at DESC LIMIT 1) AS risk_category
            FROM patients p
            LEFT JOIN health_assessments a ON a.patient_id=p.patient_id
            GROUP BY p.patient_id ORDER BY p.created_at DESC
        """)
        for r in rows:
            r['created_at'] = str(r.get('created_at',''))
            r['latest_assessment_date'] = str(r.get('latest_assessment_date','') or '')
        return rows
    result = []
    for p in patients_db.values():
        e = dict(p)
        related = [a for a in assessments_db if a['patient_id'] == p['patient_id']]
        if related:
            latest = max(related, key=lambda x: x['assessed_at'])
            e.update({'overall_risk_score': latest['overall_risk_score'],
                      'risk_category': latest['risk_category'],
                      'latest_assessment_date': latest['assessed_at'],
                      'assessment_count': len(related)})
        else:
            e.update({'overall_risk_score': None, 'risk_category': None,
                      'latest_assessment_date': None, 'assessment_count': 0})
        result.append(e)
    return sorted(result, key=lambda x: x['created_at'], reverse=True)

def db_save_assessment(pid, data, predictions, overall, category, risk_factors, recs, alerts):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    a   = {
        'patient_id': pid, 'age': data.get('age'), 'gender': data.get('gender'),
        'bmi': data.get('bmi'),
        'blood_pressure_systolic': data.get('blood_pressure_systolic'),
        'blood_pressure_diastolic': data.get('blood_pressure_diastolic'),
        'blood_glucose': data.get('blood_glucose'), 'cholesterol': data.get('cholesterol'),
        'heart_rate': data.get('heart_rate'),
        'smoking': int(bool(data.get('smoking', 0))),
        'alcohol_consumption': data.get('alcohol_consumption', 'None'),
        'physical_activity': data.get('physical_activity', 'Sedentary'),
        'diet_quality': data.get('diet_quality', 'Fair'),
        'sleep_hours': data.get('sleep_hours'), 'stress_level': data.get('stress_level'),
        'family_history_diabetes': int(bool(data.get('family_history_diabetes', 0))),
        'family_history_heart': int(bool(data.get('family_history_heart', 0))),
        'family_history_hypertension': int(bool(data.get('family_history_hypertension', 0))),
        'family_history_cancer': int(bool(data.get('family_history_cancer', 0))),
        'risk_predictions': predictions, 'overall_risk_score': overall,
        'risk_category': category, 'risk_factors': risk_factors,
        'recommendations': recs, 'alerts': alerts, 'assessed_at': now,
    }
    if DB_ENABLED:
        aid = db_execute("""
            INSERT INTO health_assessments
            (patient_id,age,gender,bmi,blood_pressure_systolic,blood_pressure_diastolic,
             blood_glucose,cholesterol,heart_rate,smoking,alcohol_consumption,
             physical_activity,diet_quality,sleep_hours,stress_level,
             family_history_diabetes,family_history_heart,family_history_hypertension,
             family_history_cancer,risk_predictions,overall_risk_score,risk_category,
             risk_factors,recommendations,alerts)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (pid, a['age'], a['gender'], a['bmi'],
              a['blood_pressure_systolic'], a['blood_pressure_diastolic'],
              a['blood_glucose'], a['cholesterol'], a['heart_rate'],
              a['smoking'], a['alcohol_consumption'], a['physical_activity'], a['diet_quality'],
              a['sleep_hours'], a['stress_level'],
              a['family_history_diabetes'], a['family_history_heart'],
              a['family_history_hypertension'], a['family_history_cancer'],
              json.dumps(predictions), overall, category,
              json.dumps(risk_factors), json.dumps(recs), json.dumps(alerts)), fetch=False)
        a['id'] = aid
    else:
        a['id'] = assessment_counter[0]
        assessments_db.append(a)
        assessment_counter[0] += 1
    return a

def db_get_history(pid):
    if DB_ENABLED:
        rows = db_execute(
            "SELECT * FROM health_assessments WHERE patient_id=%s ORDER BY assessed_at ASC", (pid,))
        for r in rows:
            for f in ['risk_predictions','risk_factors','recommendations','alerts']:
                if r.get(f) and isinstance(r[f], str):
                    try: r[f] = json.loads(r[f])
                    except: pass
            r['assessed_at'] = str(r.get('assessed_at',''))
        return rows
    return sorted([a for a in assessments_db if a['patient_id']==pid], key=lambda x: x['assessed_at'])

def db_dashboard_stats():
    if DB_ENABLED:
        total_p = db_execute("SELECT COUNT(*) as c FROM patients")[0]['c']
        total_a = db_execute("SELECT COUNT(*) as c FROM health_assessments")[0]['c']
        dist    = {'Low':0,'Moderate':0,'High':0,'Critical':0}
        for r in db_execute("SELECT risk_category,COUNT(*) as cnt FROM health_assessments GROUP BY risk_category"):
            dist[r['risk_category']] = r['cnt']
        avg  = round(float(db_execute("SELECT AVG(overall_risk_score) as a FROM health_assessments")[0]['a'] or 0), 1)
        recent = db_execute("""
            SELECT a.*, p.name AS patient_name FROM health_assessments a
            LEFT JOIN patients p ON p.patient_id=a.patient_id
            ORDER BY a.assessed_at DESC LIMIT 5
        """)
        for r in recent:
            for f in ['risk_predictions','risk_factors','recommendations','alerts']:
                if r.get(f) and isinstance(r[f], str):
                    try: r[f] = json.loads(r[f])
                    except: pass
            r['assessed_at'] = str(r.get('assessed_at',''))
        rates = {}
        for d in DISEASES:
            h = db_execute(f"SELECT COUNT(*) as c FROM health_assessments WHERE JSON_EXTRACT(risk_predictions,'$.{d}.probability')>50")[0]['c']
            rates[d] = round(h/max(total_a,1)*100, 1)
        return {'total_patients':total_p,'total_assessments':total_a,
                'risk_distribution':dist,'average_risk_score':avg,
                'disease_rates':rates,'recent_assessments':recent}
    dist = {'Low':0,'Moderate':0,'High':0,'Critical':0}
    for a in assessments_db: dist[a.get('risk_category','Low')] += 1
    avg = round(sum(a['overall_risk_score'] for a in assessments_db)/max(len(assessments_db),1),1)
    rates = {d: round(sum(1 for a in assessments_db if a.get('risk_predictions',{}).get(d,{}).get('probability',0)>50)/max(len(assessments_db),1)*100,1) for d in DISEASES}
    recent = sorted(assessments_db, key=lambda x: x['assessed_at'], reverse=True)[:5]
    for r in recent: r['patient_name'] = patients_db.get(r['patient_id'],{}).get('name','Guest')
    return {'total_patients':len(patients_db),'total_assessments':len(assessments_db),
            'risk_distribution':dist,'average_risk_score':avg,
            'disease_rates':rates,'recent_assessments':recent}

# ── ML Training ────────────────────────────────────────────────
def generate_training_data(disease, n=6000):
    np.random.seed(42)
    age=np.random.randint(18,90,n).astype(float); bmi=np.clip(np.random.normal(26,5,n),15,55)
    bp_sys=np.clip(np.random.normal(120,20,n),80,200); bp_dia=np.clip(np.random.normal(80,12,n),50,130)
    glucose=np.clip(np.random.normal(95,25,n),60,400); cholest=np.clip(np.random.normal(190,40,n),100,400)
    hr=np.clip(np.random.normal(75,12,n),45,130); smoking=np.random.randint(0,2,n).astype(float)
    alcohol=np.random.randint(0,4,n).astype(float); activity=np.random.randint(0,4,n).astype(float)
    diet=np.random.randint(0,4,n).astype(float); sleep=np.clip(np.random.normal(7,1.5,n),3,12)
    stress=np.clip(np.random.normal(5,2,n),1,10)
    fd=np.random.randint(0,2,n).astype(float); fh=np.random.randint(0,2,n).astype(float)
    fy=np.random.randint(0,2,n).astype(float); fc=np.random.randint(0,2,n).astype(float)
    X=np.column_stack([age,bmi,bp_sys,bp_dia,glucose,cholest,hr,smoking,alcohol,activity,diet,sleep,stress,fd,fh,fy,fc])
    rm={'diabetes':0.30*(glucose>126)+0.20*(bmi>30)+0.15*fd+0.10*(age>45)+0.10*(activity<2)+0.08*smoking+0.07*(diet<2),
        'heart_disease':0.25*(cholest>240)+0.20*(bp_sys>140)+0.15*smoking+0.15*fh+0.10*(age>50)+0.08*(glucose>126)+0.07*(hr>90),
        'hypertension':0.30*(bp_sys>140)+0.20*fy+0.15*(bmi>30)+0.10*(age>40)+0.10*smoking+0.10*(alcohol>2)+0.05*(stress>7),
        'stroke':0.25*(bp_sys>150)+0.20*(age>60)+0.15*fh+0.15*(cholest>240)+0.10*smoking+0.10*(glucose>126)+0.05*(stress>8),
        'obesity':0.35*(bmi>30)+0.20*(activity<1)+0.15*(diet<2)+0.10*(sleep<6)+0.10*(stress>7)+0.10*(age>40)}
    y=((rm[disease]+np.random.normal(0,0.04,n)).clip(0,1)>0.35).astype(int)
    return X, y

def train_and_save(disease):
    print(f"  Training: {disease}...")
    X,y=generate_training_data(disease)
    Xt,_,yt,_=train_test_split(X,y,test_size=0.2,random_state=42)
    sc=StandardScaler(); Xs=sc.fit_transform(Xt)
    m=GradientBoostingClassifier(n_estimators=120,learning_rate=0.08,max_depth=4,random_state=42)
    m.fit(Xs,yt)
    joblib.dump(m,  os.path.join(MODEL_DIR,f'{disease}_model.pkl'))
    joblib.dump(sc, os.path.join(MODEL_DIR,f'{disease}_scaler.pkl'))
    return m, sc

def load_models():
    for d in DISEASES:
        mp=os.path.join(MODEL_DIR,f'{d}_model.pkl'); sp=os.path.join(MODEL_DIR,f'{d}_scaler.pkl')
        if os.path.exists(mp) and os.path.exists(sp):
            MODELS[d]={'model':joblib.load(mp),'scaler':joblib.load(sp)}; print(f"  Loaded: {d}")
        else:
            m,s=train_and_save(d); MODELS[d]={'model':m,'scaler':s}

# ── Feature engineering ────────────────────────────────────────
AM={'None':0,'Occasional':1,'Moderate':2,'Heavy':3}
PM={'Sedentary':0,'Light':1,'Moderate':2,'Active':3}
DM={'Poor':0,'Fair':1,'Good':2,'Excellent':3}

def build_fv(data):
    return np.array([[float(data.get('age',35)),float(data.get('bmi',25)),
        float(data.get('blood_pressure_systolic',120)),float(data.get('blood_pressure_diastolic',80)),
        float(data.get('blood_glucose',90)),float(data.get('cholesterol',180)),
        float(data.get('heart_rate',72)),int(bool(data.get('smoking',0))),
        AM.get(data.get('alcohol_consumption','None'),0),PM.get(data.get('physical_activity','Sedentary'),0),
        DM.get(data.get('diet_quality','Fair'),1),float(data.get('sleep_hours',7)),
        float(data.get('stress_level',5)),int(bool(data.get('family_history_diabetes',0))),
        int(bool(data.get('family_history_heart',0))),int(bool(data.get('family_history_hypertension',0))),
        int(bool(data.get('family_history_cancer',0)))]])

def predict_risks(fv):
    out={}
    for d,b in MODELS.items():
        p=round(float(b['model'].predict_proba(b['scaler'].transform(fv))[0][1])*100,1)
        out[d]={'probability':p,'level':'Critical' if p>=75 else 'High' if p>=50 else 'Moderate' if p>=25 else 'Low'}
    return out

def overall_score(preds):
    w={'heart_disease':0.30,'diabetes':0.25,'hypertension':0.20,'stroke':0.15,'obesity':0.10}
    return round(sum(preds[d]['probability']*wt for d,wt in w.items()),1)

def risk_cat(s): return 'Critical' if s>=75 else 'High' if s>=50 else 'Moderate' if s>=25 else 'Low'

def gen_factors(data, preds):
    f=[]; bmi=float(data.get('bmi',25)); bps=float(data.get('blood_pressure_systolic',120))
    g=float(data.get('blood_glucose',90)); ch=float(data.get('cholesterol',180))
    age=float(data.get('age',35)); st=float(data.get('stress_level',5)); sl=float(data.get('sleep_hours',7))
    if bps>=140: f.append({'name':'High Blood Pressure','value':f'{bps} mmHg','impact':88,'level':'High'})
    elif bps>=130: f.append({'name':'Elevated BP','value':f'{bps} mmHg','impact':55,'level':'Medium'})
    if bmi>=30: f.append({'name':'Obesity (BMI)','value':f'{bmi:.1f}','impact':80,'level':'High'})
    elif bmi>=25: f.append({'name':'Overweight','value':f'{bmi:.1f}','impact':42,'level':'Medium'})
    if g>=126: f.append({'name':'High Blood Glucose','value':f'{g} mg/dL','impact':85,'level':'High'})
    elif g>=100: f.append({'name':'Elevated Glucose','value':f'{g} mg/dL','impact':48,'level':'Medium'})
    if ch>=240: f.append({'name':'High Cholesterol','value':f'{ch} mg/dL','impact':72,'level':'High'})
    elif ch>=200: f.append({'name':'Borderline Cholesterol','value':f'{ch} mg/dL','impact':38,'level':'Medium'})
    if data.get('smoking'): f.append({'name':'Active Smoker','value':'Yes','impact':78,'level':'High'})
    if age>=60: f.append({'name':'Age Risk','value':f'{int(age)} yrs','impact':65,'level':'High'})
    elif age>=45: f.append({'name':'Age Risk','value':f'{int(age)} yrs','impact':35,'level':'Medium'})
    if st>=8: f.append({'name':'High Stress','value':f'{st}/10','impact':55,'level':'Medium'})
    if sl<6: f.append({'name':'Sleep Deprivation','value':f'{sl} hrs','impact':45,'level':'Medium'})
    if data.get('alcohol_consumption')=='Heavy': f.append({'name':'Heavy Alcohol','value':'Heavy','impact':60,'level':'High'})
    if data.get('physical_activity')=='Sedentary': f.append({'name':'Inactivity','value':'Sedentary','impact':50,'level':'Medium'})
    if data.get('family_history_heart'): f.append({'name':'Family Hx — Heart','value':'Positive','impact':62,'level':'High'})
    if data.get('family_history_diabetes'): f.append({'name':'Family Hx — Diabetes','value':'Positive','impact':55,'level':'Medium'})
    f.sort(key=lambda x:x['impact'],reverse=True); return f[:8]

def gen_recs(data, preds):
    r=[]; bmi=float(data.get('bmi',25)); bps=float(data.get('blood_pressure_systolic',120))
    ch=float(data.get('cholesterol',180)); st=float(data.get('stress_level',5)); sl=float(data.get('sleep_hours',7))
    if preds.get('diabetes',{}).get('probability',0)>40: r.append({'category':'Diabetes Prevention','priority':'High','text':'Monitor blood glucose regularly. Reduce refined sugar. Schedule HbA1c every 3 months.'})
    if preds.get('heart_disease',{}).get('probability',0)>40: r.append({'category':'Cardiac Health','priority':'High','text':'Schedule ECG and lipid profile. Begin supervised cardiac exercise. Reduce saturated fat.'})
    if bps>130: r.append({'category':'Blood Pressure','priority':'High','text':'Reduce sodium to <2g/day. Daily meditation. Monitor BP twice daily and consult a cardiologist.'})
    if bmi>27: r.append({'category':'Weight Management','priority':'Medium','text':'Target 5-7% weight reduction. 150 min/week aerobic exercise with 500 kcal/day deficit.'})
    if data.get('smoking'): r.append({'category':'Smoking Cessation','priority':'High','text':'Enroll in a cessation program. Nicotine replacement therapy improves success rates.'})
    if ch>200: r.append({'category':'Cholesterol','priority':'Medium','text':'Mediterranean diet rich in omega-3. Increase soluble fibre. Re-check lipid panel in 6 weeks.'})
    if st>7: r.append({'category':'Stress Management','priority':'Medium','text':'Mindfulness-based stress reduction. Target 7-9 hours sleep. Consider CBT.'})
    if sl<6: r.append({'category':'Sleep Hygiene','priority':'Medium','text':'Consistent sleep/wake times. No screens 1 hr before bed. Evaluate for sleep apnea.'})
    if preds.get('obesity',{}).get('probability',0)>50: r.append({'category':'Obesity Risk','priority':'High','text':'Nutritionist consultation. 10,000 steps/day and progressive resistance training.'})
    if not r: r.append({'category':'General Wellness','priority':'Low','text':'Maintain healthy lifestyle. Annual full health screening recommended.'})
    return r

def gen_alerts(preds, overall):
    a=[]
    for d,info in preds.items():
        lbl=d.replace('_',' ').title()
        if info['level']=='Critical': a.append({'type':lbl,'severity':'Critical','message':f"CRITICAL: {lbl} risk {info['probability']}%. Immediate medical consultation required."})
        elif info['level']=='High': a.append({'type':lbl,'severity':'Warning','message':f"HIGH RISK: {lbl} at {info['probability']}%. See a physician within 2 weeks."})
    if overall>60: a.append({'type':'Overall Risk','severity':'Critical','message':f'Overall score {overall}% — multi-specialist consultation strongly advised.'})
    return a

# ── CORS ───────────────────────────────────────────────────────
@app.after_request
def cors(r):
    r.headers['Access-Control-Allow-Origin']  = '*'
    r.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    r.headers['Access-Control-Allow-Methods'] = 'GET,POST,PUT,DELETE,OPTIONS'
    return r

@app.route('/api/<path:_>', methods=['OPTIONS'])
def opt(_): return '', 204

# ── Frontend ───────────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def frontend(path):
    if path and os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, 'index.html')

# ── API routes ─────────────────────────────────────────────────
@app.route('/api/health')
def health():
    tp = db_execute("SELECT COUNT(*) as c FROM patients")[0]['c'] if DB_ENABLED else len(patients_db)
    ta = db_execute("SELECT COUNT(*) as c FROM health_assessments")[0]['c'] if DB_ENABLED else len(assessments_db)
    return jsonify({'status':'ok','version':'5.0','db_connected':DB_ENABLED,
                    'storage':'mysql' if DB_ENABLED else 'in-memory',
                    'models_loaded':list(MODELS.keys()),
                    'total_patients':tp,'total_assessments':ta,
                    'timestamp':datetime.now().isoformat()})

@app.route('/api/patients', methods=['GET'])
def get_patients():
    d = db_get_all_patients()
    return jsonify({'success':True,'data':d,'total':len(d)})

@app.route('/api/patients', methods=['POST'])
def create_patient():
    data = request.get_json() or {}
    if not data.get('name'): return jsonify({'success':False,'message':'Name required'}),400
    p = db_create_patient(data)
    return jsonify({'success':True,'data':p,'message':f"Patient {p['patient_id']} registered"})

@app.route('/api/patients/<pid>', methods=['GET'])
def get_patient(pid):
    p = db_get_patient(pid)
    if not p: return jsonify({'success':False,'message':'Not found'}),404
    h = db_get_history(pid)
    return jsonify({'success':True,'data':{**dict(p),'assessments':h,'assessment_count':len(h)}})

@app.route('/api/patients/<pid>/history')
def patient_history(pid):
    h = db_get_history(pid)
    return jsonify({'success':True,'data':h,'total':len(h)})

@app.route('/api/assess', methods=['POST'])
def assess():
    data = request.get_json() or {}
    pid  = data.get('patient_id')
    if pid and not db_get_patient(pid): return jsonify({'success':False,'message':f'{pid} not found'}),404
    if not MODELS: return jsonify({'success':False,'message':'Models not loaded'}),503
    try:
        fv    = build_fv(data)
        preds = predict_risks(fv)
        ov    = overall_score(preds)
        cat   = risk_cat(ov)
        rf    = gen_factors(data, preds)
        recs  = gen_recs(data, preds)
        alts  = gen_alerts(preds, ov)
        a     = db_save_assessment(pid or 'GUEST', data, preds, ov, cat, rf, recs, alts)
        return jsonify({'success':True,'data':{'assessment':a,'predictions':preds,
            'overall_risk_score':ov,'risk_category':cat,'risk_factors':rf,
            'recommendations':recs,'alerts':alts}})
    except Exception as e:
        return jsonify({'success':False,'message':str(e)}),500

@app.route('/api/dashboard/stats')
def dashboard(): return jsonify({'success':True,'data':db_dashboard_stats()})

@app.route('/api/analytics/population')
def analytics():
    ag={'18-29':[],'30-44':[],'45-59':[],'60+':[]}; gg={}
    rows = db_execute("SELECT age,gender,overall_risk_score FROM health_assessments") if DB_ENABLED \
           else [{'age':a.get('age',0),'gender':a.get('gender','Unknown'),'overall_risk_score':a.get('overall_risk_score',0)} for a in assessments_db]
    for r in rows:
        age=int(r.get('age') or 0); sc=float(r.get('overall_risk_score') or 0)
        if age<30: ag['18-29'].append(sc)
        elif age<45: ag['30-44'].append(sc)
        elif age<60: ag['45-59'].append(sc)
        else: ag['60+'].append(sc)
        gg.setdefault(r.get('gender','Unknown'),[]).append(sc)
    return jsonify({'success':True,'data':{
        'by_age':[{'age_group':g,'avg_risk':round(sum(v)/len(v),1) if v else 0,'count':len(v)} for g,v in ag.items()],
        'by_gender':[{'gender':g,'avg_risk':round(sum(v)/len(v),1) if v else 0,'count':len(v)} for g,v in gg.items()]}})

# ── Startup ────────────────────────────────────────────────────
print("="*55)
print("  Predictive Health Risk Analysis  v5.0")
print("="*55)
print("\n[1] Connecting to local MySQL...")
init_db()
print(f"    Storage: {'MySQL  (persistent)' if DB_ENABLED else 'In-Memory (temporary)'}")
print("\n[2] Loading ML models...")
load_models()
print(f"    Ready: {list(MODELS.keys())}")
print("\n[3] Server ready.\n")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"    http://localhost:{port}\n")
    app.run(debug=True, host='0.0.0.0', port=port)
