import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard — Cardiovascular Disease Prediction",
    page_icon="🫀",
    layout="wide",
)

plt.style.use('seaborn-v0_8-whitegrid')
RANDOM_STATE = 42

# ─── Load & Cache Data ───────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
               'thalach','exang','oldpeak','slope','ca','thal','target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    df.fillna(df.median(numeric_only=True), inplace=True)

    categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    le = LabelEncoder()
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train_sc, X_test_sc, _, _ = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    return df, df_encoded, X, y, X_scaled, scaler, le, categorical_cols, \
           X_train, X_test, X_train_sc, X_test_sc, y_train, y_test

@st.cache_resource
def train_models(X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, X, X_scaled, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_sc, y_train)

    svm = SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_sc, y_train)

    fitted = {
        'Random Forest':       (rf,  X_test,    X),
        'Logistic Regression': (lr,  X_test_sc, X_scaled),
        'SVM':                 (svm, X_test_sc, X_scaled),
    }
    test_results = {}
    kfold_results = {}

    for name, (model, X_t, X_full) in fitted.items():
        y_pred  = model.predict(X_t)
        y_proba = model.predict_proba(X_t)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        test_results[name] = {'y_pred': y_pred, 'y_proba': y_proba,
                               'accuracy': acc, 'roc_auc': auc}
        scores = cross_val_score(model, X_full, y, cv=kf, scoring='accuracy')
        kfold_results[name] = scores

    return rf, lr, svm, test_results, kfold_results

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=80)
st.sidebar.title("HeartGuard 🫀")
st.sidebar.markdown("Cardiovascular Disease Prediction using ML")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "📊 Data Explorer",
    "🤖 Model Performance",
    "🩺 Patient Prediction",
    "📂 Bulk Scanner"
])

# ─── Load Data ───────────────────────────────────────────────────────────────
with st.spinner("Loading dataset and training models…"):
    (df, df_encoded, X, y, X_scaled, scaler, le, categorical_cols,
     X_train, X_test, X_train_sc, X_test_sc, y_train, y_test) = load_and_prepare_data()

    rf, lr, svm, test_results, kfold_results = train_models(
        X_train, X_test, X_train_sc, X_test_sc, y_train, y_test, X, X_scaled, y)

# ════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🫀 HeartGuard — Cardiovascular Disease Prediction")
    st.markdown("""
    > **Predict the likelihood of heart disease onset using machine learning.**

    This app is built on the **UCI Heart Disease Dataset** (Cleveland subset — 303 records, 14 attributes).
    Three models are trained and compared:
    - 🌲 **Random Forest**
    - 📈 **Logistic Regression**
    - 🔵 **Support Vector Machine (SVM)**
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📋 Records", df.shape[0])
    col2.metric("🔢 Features", df.shape[1] - 1)
    best_model = max(test_results, key=lambda m: test_results[m]['accuracy'])
    col3.metric("🏆 Best Accuracy", f"{test_results[best_model]['accuracy']*100:.1f}%", best_model)
    col4.metric("📉 Disease Cases", int(df['target'].sum()))

    st.divider()
    st.subheader("Feature Glossary")
    glossary = {
        "age": "Age in years",
        "sex": "Sex (1=Male, 0=Female)",
        "cp": "Chest pain type (0–3)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl (1=True)",
        "restecg": "Resting ECG results (0–2)",
        "thalach": "Maximum heart rate achieved",
        "exang": "Exercise-induced angina (1=Yes)",
        "oldpeak": "ST depression induced by exercise",
        "slope": "Slope of peak exercise ST segment",
        "ca": "Number of major vessels (0–3) coloured by fluoroscopy",
        "thal": "Thalassemia (1=Normal, 2=Fixed defect, 3=Reversible defect)",
        "target": "Heart disease present (1=Yes, 0=No)",
    }
    st.dataframe(pd.DataFrame(glossary.items(), columns=["Feature", "Description"]),
                 use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(["Raw Data", "Target Distribution",
                                       "Feature Distributions", "Correlation"])

    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.success("✅ No missing values after median imputation.")
        else:
            st.warning(f"Missing values found: {miss.to_dict()}")

    with tab2:
        st.subheader("Target Class Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        tc = df['target'].value_counts()
        axes[0].bar(['No Disease (0)', 'Disease (1)'], tc.values,
                    color=['#2ecc71', '#e74c3c'], edgecolor='white', linewidth=1.5)
        axes[0].set_title('Count', fontweight='bold')
        for i, v in enumerate(tc.values):
            axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
        axes[1].pie(tc.values, labels=['No Disease', 'Disease'],
                    autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'],
                    startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        axes[1].set_title('Proportion', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Continuous Features by Disease Status")
        cont = ['age','trestbps','chol','thalach','oldpeak']
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for i, col in enumerate(cont):
            for label, color, name in zip([0,1],['#2ecc71','#e74c3c'],['No Disease','Disease']):
                axes[i].hist(df[df['target']==label][col].dropna(),
                             alpha=0.6, color=color, label=name, bins=20, edgecolor='white')
            axes[i].set_title(col.upper(), fontweight='bold')
            axes[i].legend(fontsize=7)
        axes[-1].axis('off')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Categorical Features by Disease Status")
        cat_features = ['sex','cp','fbs','restecg','exang','slope']
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        axes = axes.flatten()
        for i, col in enumerate(cat_features):
            grp = df.groupby([col,'target']).size().unstack(fill_value=0)
            grp.plot(kind='bar', ax=axes[i], color=['#2ecc71','#e74c3c'],
                     edgecolor='white', linewidth=1)
            axes[i].set_title(col.upper(), fontweight='bold')
            axes[i].legend(['No Disease','Disease'], fontsize=7)
            axes[i].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.subheader("Feature Correlation Matrix")
        fig, ax = plt.subplots(figsize=(11, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
                    center=0, linewidths=0.5, ax=ax, annot_kws={'size': 8})
        ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "K-Fold CV",
                                       "Confusion Matrices", "ROC Curves"])

    with tab1:
        st.subheader("Model Comparison Summary")
        comparison = pd.DataFrame({
            'Model': list(test_results.keys()),
            'Test Accuracy': [f"{r['accuracy']*100:.2f}%" for r in test_results.values()],
            'ROC-AUC': [f"{r['roc_auc']:.4f}" for r in test_results.values()],
            'CV Mean Acc': [f"{kfold_results[m].mean()*100:.2f}%" for m in test_results],
            'CV Std': [f"{kfold_results[m].std():.4f}" for m in test_results],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.subheader("Performance Bar Chart")
        comp_num = pd.DataFrame({
            'Test Accuracy': [r['accuracy'] for r in test_results.values()],
            'ROC-AUC':       [r['roc_auc']  for r in test_results.values()],
            'CV Mean Acc':   [kfold_results[m].mean() for m in test_results],
        }, index=list(test_results.keys()))
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(comp_num.index))
        w = 0.25
        colors_bar = ['#3498db','#e74c3c','#2ecc71']
        for i, (col, c) in enumerate(zip(comp_num.columns, colors_bar)):
            bars = ax.bar(x + i*w, comp_num[col], w, label=col, color=c, alpha=0.85, edgecolor='white')
            for bar in bars:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x + w)
        ax.set_xticklabels(comp_num.index, fontsize=11)
        ax.set_ylim(0.6, 1.05)
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Feature Importances — Random Forest")
        feature_names = X.columns.tolist()
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(feature_names)), importances[indices],
                      color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_names))),
                      edgecolor='white', linewidth=1.2)
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=40, ha='right')
        ax.set_title('Feature Importances — Random Forest', fontsize=13, fontweight='bold')
        ax.set_ylabel('Importance Score')
        for bar, imp in zip(bars, importances[indices]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{imp:.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("10-Fold Cross Validation Accuracy")
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#3498db','#e67e22','#9b59b6']
        for i, (name, scores) in enumerate(kfold_results.items()):
            ax.plot(range(1,11), scores, marker='o', label=name, color=colors[i], linewidth=2)
        ax.set_title('10-Fold CV Accuracy per Model', fontsize=13, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(range(1,11))
        ax.legend()
        ax.set_ylim(0.6, 1.05)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Confusion Matrices — Test Set")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (name, res) in zip(axes, test_results.items()):
            cm = confusion_matrix(y_test, res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['No Disease','Disease'],
                        yticklabels=['No Disease','Disease'],
                        linewidths=1, linecolor='white')
            ax.set_title(f'{name}\nAcc: {res["accuracy"]:.3f}', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#3498db','#e67e22','#9b59b6']
        for (name, res), color in zip(test_results.items(), colors):
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})",
                    color=color, linewidth=2)
        ax.plot([0,1],[0,1],'k--', linewidth=1, label='Random Classifier')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PATIENT PREDICTION
# ════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Patient Prediction":
    st.title("🩺 Patient Heart Disease Risk Predictor")
    st.markdown("Enter the patient's clinical measurements below to predict their cardiovascular risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        age      = st.slider("Age", 20, 80, 54)
        sex      = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
        cp       = st.selectbox("Chest Pain Type",
                                [(0,"Typical Angina"),(1,"Atypical Angina"),
                                 (2,"Non-Anginal"),(3,"Asymptomatic")],
                                format_func=lambda x: x[1])[0]
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 130)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 250)

    with col2:
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                               [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])[0]
        restecg = st.selectbox("Resting ECG",
                               [(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")],
                               format_func=lambda x: x[1])[0]
        thalach = st.slider("Max Heart Rate Achieved", 60, 220, 147)
        exang   = st.selectbox("Exercise-Induced Angina",
                               [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])[0]
        oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.4, step=0.1)

    with col3:
        slope = st.selectbox("Slope of ST Segment",
                             [(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")],
                             format_func=lambda x: x[1])[0]
        ca    = st.selectbox("Major Vessels (fluoroscopy)", [0, 1, 2, 3])
        thal  = st.selectbox("Thalassemia",
                             [(1,"Normal"),(2,"Fixed Defect"),(3,"Reversible Defect")],
                             format_func=lambda x: x[1])[0]
        model_choice = st.selectbox("Model to Use",
                                    ["Random Forest", "Logistic Regression", "SVM"])

    if st.button("🔍 Predict Risk", use_container_width=True, type="primary"):
        sample = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])

        # Encode categorical — apply same LabelEncoder per column
        cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
        sample_enc = sample.copy()
        for col in cat_cols:
            sample_enc[col] = LabelEncoder().fit_transform(sample_enc[col].astype(str))

        sample_scaled = scaler.transform(sample_enc)

        model_map = {
            "Random Forest":       (rf,  sample_enc),
            "Logistic Regression": (lr,  sample_scaled),
            "SVM":                 (svm, sample_scaled),
        }
        chosen_model, chosen_input = model_map[model_choice]

        prediction  = chosen_model.predict(chosen_input)[0]
        probability = chosen_model.predict_proba(chosen_input)[0][1] * 100

        st.divider()
        st.subheader("📋 Prediction Result")
        rcol1, rcol2 = st.columns(2)

        with rcol1:
            if prediction == 1:
                st.error(f"⚠️  **Heart Disease Detected**")
            else:
                st.success(f"✅  **No Heart Disease Detected**")
            st.metric("Risk Probability", f"{probability:.1f}%")
            st.metric("Model Used", model_choice)

        with rcol2:
            # Gauge-style bar
            fig, ax = plt.subplots(figsize=(5, 1.5))
            color = '#e74c3c' if probability >= 50 else '#2ecc71'
            ax.barh(['Risk'], [probability], color=color, height=0.5)
            ax.barh(['Risk'], [100 - probability], left=[probability],
                    color='#ecf0f1', height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel('Risk %')
            ax.set_title(f'Risk Score: {probability:.1f}%', fontweight='bold')
            ax.axvline(50, color='gray', linestyle='--', linewidth=1)
            ax.text(50, 0, '50%', ha='center', va='bottom', fontsize=8, color='gray')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.info("""
        **Disclaimer:** This tool is for educational/research purposes only. 
        It is not a substitute for professional medical diagnosis or advice.
        """)

# ════════════════════════════════════════════════════════════════════════════
# BULK SCANNER
# ════════════════════════════════════════════════════════════════════════════
elif page == "📂 Bulk Scanner":
    st.title("📂 Bulk Patient Scanner")
    st.markdown("Upload a **CSV file** with multiple patients to get batch heart disease predictions instantly.")

    # ── Template download ──────────────────────────────────────────────────
    with st.expander("📋 How to use & CSV Template", expanded=True):
        st.markdown("""
        **Steps:**
        1. Download the template below or prepare a CSV with the required columns.
        2. Fill in patient data — one patient per row.
        3. Upload the file, choose a model, and click **Run Scan**.
        4. View results, filter, and download the prediction report.

        **Required columns:** `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

        **Value guide:** `sex` 0=Female/1=Male | `cp` 0–3 | `fbs` 0/1 | `restecg` 0–2 | `exang` 0/1 | `slope` 0–2 | `ca` 0–3 | `thal` 1–3
        """)
        template = pd.DataFrame([
            {'age':54,'sex':1,'cp':2,'trestbps':130,'chol':250,'fbs':0,'restecg':1,'thalach':147,'exang':0,'oldpeak':1.4,'slope':1,'ca':1,'thal':2},
            {'age':67,'sex':1,'cp':3,'trestbps':160,'chol':286,'fbs':0,'restecg':2,'thalach':108,'exang':1,'oldpeak':1.5,'slope':1,'ca':3,'thal':2},
            {'age':41,'sex':0,'cp':1,'trestbps':130,'chol':204,'fbs':0,'restecg':0,'thalach':172,'exang':0,'oldpeak':1.4,'slope':0,'ca':0,'thal':2},
            {'age':57,'sex':0,'cp':0,'trestbps':120,'chol':354,'fbs':0,'restecg':1,'thalach':163,'exang':1,'oldpeak':0.6,'slope':0,'ca':0,'thal':2},
            {'age':63,'sex':1,'cp':3,'trestbps':145,'chol':233,'fbs':1,'restecg':0,'thalach':150,'exang':0,'oldpeak':2.3,'slope':2,'ca':0,'thal':1},
        ])
        st.download_button(
            label="⬇️ Download Sample Template CSV",
            data=template.to_csv(index=False).encode('utf-8'),
            file_name="heartguard_template.csv",
            mime="text/csv",
        )

    st.divider()

    ucol1, ucol2 = st.columns([3, 1])
    with ucol1:
        uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"],
                                         help="Must contain all 13 required feature columns.")
    with ucol2:
        bulk_model = st.selectbox("Model", ["Random Forest", "Logistic Regression", "SVM"], key="bulk_model")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            required_cols = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
            missing_cols = [c for c in required_cols if c not in raw_df.columns]
            if missing_cols:
                st.error(f"❌ Missing columns: `{', '.join(missing_cols)}`")
                st.stop()

            st.success(f"✅ File loaded — **{len(raw_df)} patients** detected.")
            with st.expander("🔍 Preview Uploaded Data"):
                st.dataframe(raw_df.head(10), use_container_width=True)
                if len(raw_df) > 10:
                    st.caption(f"Showing first 10 of {len(raw_df)} rows.")

            # Pre-process
            input_df = raw_df[required_cols].copy()
            input_df.fillna(input_df.median(numeric_only=True), inplace=True)
            input_enc = input_df.copy()
            for col in ['sex','cp','fbs','restecg','exang','slope','ca','thal']:
                input_enc[col] = LabelEncoder().fit_transform(input_enc[col].astype(str))
            input_scaled = scaler.transform(input_enc)

            # Predict
            model_map = {
                "Random Forest":       (rf,  input_enc),
                "Logistic Regression": (lr,  input_scaled),
                "SVM":                 (svm, input_scaled),
            }
            chosen_model, chosen_input = model_map[bulk_model]
            predictions   = chosen_model.predict(chosen_input)
            probabilities = chosen_model.predict_proba(chosen_input)[:, 1] * 100

            # Build results dataframe
            results_df = raw_df.copy()
            results_df['Prediction']  = predictions
            results_df['Diagnosis']   = results_df['Prediction'].map({0:'✅ No Disease', 1:'⚠️ Heart Disease'})
            results_df['Risk (%)']    = probabilities.round(1)
            results_df['Risk Level']  = pd.cut(probabilities, bins=[0,30,60,100],
                                               labels=['🟢 Low','🟡 Medium','🔴 High'])
            results_df['Model Used']  = bulk_model

            # Summary metrics
            st.divider()
            st.subheader("📊 Batch Results Summary")
            n_total   = len(results_df)
            n_disease = int((predictions == 1).sum())
            n_healthy = int((predictions == 0).sum())
            avg_risk  = probabilities.mean()
            high_risk = int((probabilities >= 60).sum())

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("👥 Total Patients", n_total)
            mc2.metric("⚠️ Disease Detected", n_disease,
                       delta=f"{n_disease/n_total*100:.1f}%", delta_color="inverse")
            mc3.metric("✅ No Disease", n_healthy)
            mc4.metric("📊 Avg Risk Score", f"{avg_risk:.1f}%")
            mc5.metric("🔴 High Risk (≥60%)", high_risk)

            # Charts
            ch1, ch2 = st.columns(2)
            with ch1:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie([n_healthy, n_disease], labels=['No Disease','Heart Disease'],
                       autopct='%1.1f%%', colors=['#2ecc71','#e74c3c'],
                       startangle=90, wedgeprops={'edgecolor':'white','linewidth':2})
                ax.set_title('Diagnosis Distribution', fontweight='bold', fontsize=12)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            with ch2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.hist(probabilities, bins=20, color='#3498db', edgecolor='white', linewidth=1.2, alpha=0.85)
                ax.axvline(50, color='#e74c3c', linestyle='--', linewidth=1.5, label='50% threshold')
                ax.axvline(avg_risk, color='#f39c12', linestyle='--', linewidth=1.5, label=f'Mean: {avg_risk:.1f}%')
                ax.set_xlabel('Risk Score (%)'); ax.set_ylabel('Patients')
                ax.set_title('Risk Score Distribution', fontweight='bold', fontsize=12)
                ax.legend(fontsize=9)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            fig, ax = plt.subplots(figsize=(10, 2.5))
            risk_order  = ['🟢 Low','🟡 Medium','🔴 High']
            risk_counts = results_df['Risk Level'].value_counts()
            risk_vals   = [risk_counts.get(r, 0) for r in risk_order]
            bars = ax.barh(risk_order, risk_vals,
                           color=['#2ecc71','#f39c12','#e74c3c'], edgecolor='white', linewidth=1.5, height=0.5)
            for bar, val in zip(bars, risk_vals):
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                        str(val), va='center', fontweight='bold')
            ax.set_xlabel('Number of Patients')
            ax.set_title('Risk Level Breakdown', fontweight='bold', fontsize=12)
            ax.set_xlim(0, max(risk_vals) * 1.2 + 1)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Full results table with filters
            st.divider()
            st.subheader("📋 Full Patient Results")
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                filter_diag = st.multiselect("Filter by Diagnosis",
                    ['✅ No Disease','⚠️ Heart Disease'],
                    default=['✅ No Disease','⚠️ Heart Disease'])
            with fcol2:
                filter_risk = st.multiselect("Filter by Risk Level",
                    ['🟢 Low','🟡 Medium','🔴 High'],
                    default=['🟢 Low','🟡 Medium','🔴 High'])

            filtered = results_df[
                results_df['Diagnosis'].isin(filter_diag) &
                results_df['Risk Level'].isin(filter_risk)
            ]
            display_cols = required_cols + ['Diagnosis','Risk (%)','Risk Level']
            st.dataframe(
                filtered[display_cols].reset_index(drop=True),
                use_container_width=True,
                column_config={
                    "Risk (%)": st.column_config.ProgressColumn(
                        "Risk (%)", min_value=0, max_value=100, format="%.1f%%"),
                }
            )
            st.caption(f"Showing {len(filtered)} of {n_total} patients.")

            # Download
            st.divider()
            export_df = results_df[display_cols].copy()
            export_df['Diagnosis']  = export_df['Diagnosis'].str.replace(r'[^\w\s%]','', regex=True).str.strip()
            export_df['Risk Level'] = export_df['Risk Level'].str.replace(r'[^\w\s]','', regex=True).str.strip()
            st.download_button(
                label="⬇️ Download Full Prediction Report (CSV)",
                data=export_df.to_csv(index=False).encode('utf-8'),
                file_name="heartguard_bulk_predictions.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
            st.info("**Disclaimer:** For educational/research purposes only. Not a substitute for professional medical advice.")

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Please ensure your CSV matches the required format. Download the template above for reference.")

    else:
        st.info("👆 Upload a CSV file above to begin bulk scanning.")
        st.markdown("""
        #### What the Bulk Scanner does:
        - 📁 **Processes any number of patients** in one go from a single CSV
        - 🤖 **Runs predictions** using Random Forest, Logistic Regression, or SVM
        - 📊 **Summary stats**: total patients, disease count, avg risk score, high-risk count
        - 📈 **Visual breakdowns**: diagnosis pie chart, risk distribution histogram, risk level bar chart
        - 🔎 **Filterable results table** with colour-coded risk progress bar
        - ⬇️ **Downloadable CSV report** for clinical or research use
        """)