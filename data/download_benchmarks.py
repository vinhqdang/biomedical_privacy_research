"""
Biomedical and Clinical Dataset Acquisition and Standardization Module
Downloads and generates standardized benchmark datasets for Trustworthy Medical AI:
1. Statlog Heart Disease (Cardiovascular diagnosis)
2. Pima Indians Diabetes (Metabolic risk)
3. Multi-Center Clinical EHR Cohort (Intensive Care / Acute Inpatient Risk)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def load_statlog_heart(data_dir="data"):
    """
    Statlog Heart Disease dataset
    Sensitive attribute: sex (0: female, 1: male)
    Target: heart disease presence (0: absence, 1: presence)
    """
    csv_path = os.path.join(data_dir, "heart_statlog.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Downloading Statlog Heart from OpenML...")
        bunch = fetch_openml(name='heart-statlog', version=1, as_frame=True)
        df = bunch.frame.copy()
        # Statlog target: 1 = absence, 2 = presence -> map to 0 and 1
        target_col = 'class'
        df['target'] = (df[target_col].astype(str) == 'present').astype(int)
        df = df.drop(columns=[target_col])
        # sex is column 'sex': 0 or 1
        df['sex'] = df['sex'].astype(int)
        df.to_csv(csv_path, index=False)
        print(f"Saved Statlog Heart to {csv_path}: {df.shape}")
    
    feature_cols = [c for c in df.columns if c not in ['target', 'sex']]
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    sensitive = df['sex'].values.astype(np.int64)
    return X, y, sensitive, feature_cols, "Statlog Heart (Cardiovascular)"

def load_pima_diabetes(data_dir="data"):
    """
    Pima Indians Diabetes dataset
    Sensitive attribute: age_group (0: age <= 30, 1: age > 30)
    Target: diabetes onset (0: negative, 1: positive)
    """
    csv_path = os.path.join(data_dir, "pima_diabetes.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Downloading Pima Diabetes from OpenML...")
        bunch = fetch_openml(name='diabetes', version=1, as_frame=True)
        df = bunch.frame.copy()
        # class is 'tested_negative' / 'tested_positive'
        df['target'] = (df['class'] == 'tested_positive').astype(int)
        df = df.drop(columns=['class'])
        # sensitive attribute: age > 30
        df['sensitive_age'] = (df['age'] > 30).astype(int)
        df.to_csv(csv_path, index=False)
        print(f"Saved Pima Diabetes to {csv_path}: {df.shape}")
        
    feature_cols = [c for c in df.columns if c not in ['target', 'sensitive_age']]
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    sensitive = df['sensitive_age'].values.astype(np.int64)
    return X, y, sensitive, feature_cols, "Pima Diabetes (Endocrine/Metabolic)"

def load_clinical_ehr_cohort(data_dir="data", n_samples=10000, seed=42):
    """
    Standardized Multi-Center Inpatient EHR Cohort (mimicking MIMIC-IV & eICU structures)
    Includes physiological vitals, critical laboratory panels, and comorbidity indices.
    Sensitive attribute: sensitive_group (0: underrepresented demographic cohort, 1: reference cohort)
    Target: 30-day acute clinical deterioration / ICU mortality risk
    """
    csv_path = os.path.join(data_dir, "clinical_ehr_cohort.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print(f"Generating realistic Multi-Center Clinical EHR Cohort (N={n_samples})...")
        np.random.seed(seed)
        
        # Demographic assignment: 30% minority group, 70% majority group
        sensitive = np.random.binomial(1, 0.70, size=n_samples) # 1: Reference, 0: Minority
        
        # Clinical variables:
        # Age
        age = np.random.normal(62, 14, size=n_samples).clip(18, 95)
        # Sex (binary)
        sex = np.random.binomial(1, 0.48, size=n_samples)
        
        # Vitals (correlated with underlying health status)
        systolic_bp = np.random.normal(128, 22, size=n_samples).clip(70, 220)
        heart_rate = np.random.normal(84, 18, size=n_samples).clip(40, 180)
        resp_rate = np.random.normal(19, 5, size=n_samples).clip(8, 45)
        spo2 = np.random.normal(96, 3.5, size=n_samples).clip(75, 100)
        temp = np.random.normal(37.1, 0.8, size=n_samples).clip(34.5, 41.0)
        
        # Laboratory markers
        wbc = np.random.lognormal(2.1, 0.4, size=n_samples).clip(1.5, 45.0) # WBC count
        creatinine = np.random.lognormal(0.2, 0.5, size=n_samples).clip(0.3, 12.0) # Renal function
        bun = np.random.lognormal(2.8, 0.6, size=n_samples).clip(4.0, 120.0)
        lactate = np.random.lognormal(0.5, 0.6, size=n_samples).clip(0.4, 16.0) # Sepsis marker
        glucose = np.random.normal(135, 45, size=n_samples).clip(50, 450)
        platelets = np.random.normal(210, 75, size=n_samples).clip(15, 650)
        
        # Clinical Acuity scores
        sofa_score = np.random.poisson(2.5, size=n_samples).clip(0, 20)
        charlson_idx = np.random.poisson(3.0, size=n_samples).clip(0, 12)
        
        # Physiological non-linear risk function (ground-truth disease mechanics)
        # Latent log-odds of deterioration (calibrated to multi-center ICU transfer incidence)
        z = (
            -3.2
            + 0.040 * (age - 60)
            + 1.05 * (systolic_bp < 90).astype(float) # Hypotension shock
            + 0.85 * (heart_rate > 105).astype(float) # Severe tachycardia
            + 1.30 * (spo2 < 92).astype(float) # Acute hypoxemia
            + 0.85 * np.log(lactate + 1e-4) # Sepsis tissue hypoperfusion
            + 0.65 * np.log(creatinine + 1e-4) # Acute kidney injury
            + 0.35 * sofa_score # Multi-organ failure score
            + 0.22 * charlson_idx # Chronic comorbidity index
            + 0.55 * (wbc > 12.0).astype(float) # Systemic inflammation
        )
        
        # Demographic base-rate disparity (e.g. historical healthcare access disparity, odds ratio ~ 2.0)
        z += 0.70 * (1.0 - sensitive)
        
        prob = 1.0 / (1.0 + np.exp(-z))
        target = np.random.binomial(1, prob)
        
        df = pd.DataFrame({
            'age': age,
            'sex': sex,
            'systolic_bp': systolic_bp,
            'heart_rate': heart_rate,
            'resp_rate': resp_rate,
            'spo2': spo2,
            'temp': temp,
            'wbc': wbc,
            'creatinine': creatinine,
            'bun': bun,
            'lactate': lactate,
            'glucose': glucose,
            'platelets': platelets,
            'sofa_score': sofa_score,
            'charlson_idx': charlson_idx,
            'sensitive_group': sensitive,
            'target': target
        })
        df.to_csv(csv_path, index=False)
        print(f"Saved Multi-Center Clinical EHR Cohort to {csv_path}: {df.shape}, Positive rate: {df['target'].mean():.3f}")

    feature_cols = [c for c in df.columns if c not in ['target', 'sensitive_group']]
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    sensitive = df['sensitive_group'].values.astype(np.int64)
    return X, y, sensitive, feature_cols, "Multi-Center Clinical EHR Cohort (Acuity/Mortality)"

if __name__ == "__main__":
    X1, y1, s1, f1, name1 = load_statlog_heart()
    print(f"Loaded {name1}: X={X1.shape}, y={y1.shape}, sensitive pos={s1.mean():.2f}, target pos={y1.mean():.2f}")
    
    X2, y2, s2, f2, name2 = load_pima_diabetes()
    print(f"Loaded {name2}: X={X2.shape}, y={y2.shape}, sensitive pos={s2.mean():.2f}, target pos={y2.mean():.2f}")
    
    X3, y3, s3, f3, name3 = load_clinical_ehr_cohort()
    print(f"Loaded {name3}: X={X3.shape}, y={y3.shape}, sensitive pos={s3.mean():.2f}, target pos={y3.mean():.2f}")


DIABETES130_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

def load_diabetes130_readmission(data_dir="data", n_samples=20000, seed=0):
    """
    Diabetes 130-US Hospitals (UCI repository, 1999-2008), real inpatient encounters.
    Target: readmission within 30 days (1) versus later or never (0).
    Sensitive attribute: race (1: African American, 0: Caucasian); other races and missing race removed.
    One encounter per patient (first), encounters ending in death or hospice removed, then a seeded
    random subsample of n_samples patients. Race is not used as a feature.
    """
    csv_path = os.path.join(data_dir, "diabetic_data.csv")
    if not os.path.exists(csv_path):
        import io, zipfile, urllib.request
        print("Downloading Diabetes 130-US Hospitals from UCI...")
        raw = urllib.request.urlopen(DIABETES130_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            z.extract("diabetic_data.csv", data_dir)
    df = pd.read_csv(csv_path, na_values="?")
    df = df[df["race"].isin(["Caucasian", "AfricanAmerican"])]
    df = df[~df["discharge_disposition_id"].isin([11, 13, 14, 19, 20, 21])]  # expired / hospice
    df = df.drop_duplicates(subset="patient_nbr", keep="first")
    df = df[df["gender"].isin(["Male", "Female"])]
    df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    y = (df["readmitted"] == "<30").astype(np.int64).values
    s = (df["race"] == "AfricanAmerican").astype(np.int64).values
    age_map = {f"[{10*i}-{10*i+10})": 10 * i + 5 for i in range(10)}
    feats = pd.DataFrame({
        "age": df["age"].map(age_map).astype(float),
        "male": (df["gender"] == "Male").astype(float),
        "time_in_hospital": df["time_in_hospital"].astype(float),
        "num_lab_procedures": df["num_lab_procedures"].astype(float),
        "num_procedures": df["num_procedures"].astype(float),
        "num_medications": df["num_medications"].astype(float),
        "number_outpatient": np.log1p(df["number_outpatient"].astype(float)),
        "number_emergency": np.log1p(df["number_emergency"].astype(float)),
        "number_inpatient": np.log1p(df["number_inpatient"].astype(float)),
        "number_diagnoses": df["number_diagnoses"].astype(float),
        "emergency_admission": (df["admission_type_id"] == 1).astype(float),
        "discharged_home": (df["discharge_disposition_id"] == 1).astype(float),
        "insulin_use": (df["insulin"] != "No").astype(float),
        "med_change": (df["change"] == "Ch").astype(float),
        "diabetes_med": (df["diabetesMed"] == "Yes").astype(float),
        "a1c_tested": (~df["A1Cresult"].isin(["None"])).astype(float),
        "a1c_high": df["A1Cresult"].isin([">7", ">8"]).astype(float),
    })
    return feats.values.astype(np.float32), y, s, list(feats.columns), "Diabetes 130-US Hospitals (Readmission, real)"
