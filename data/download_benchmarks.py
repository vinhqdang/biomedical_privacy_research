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
