# train.py
# 100% ML Recommendation Engine (Local VS Code Version)
# ----------------------------------------------------

import os, re, joblib, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

# ---------------------------
# Step 1 - Folder Setup
# ---------------------------
os.makedirs('models', exist_ok=True)
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('embeddings', exist_ok=True)

# ---------------------------
# Step 2 - Load Dataset
# ---------------------------
DATA_PATH = 'data/Labour_Expanded_2000.xlsx'

print("📥 Reading dataset from:", DATA_PATH)
DF = pd.read_excel(DATA_PATH)
DF.columns = [c.strip() for c in DF.columns]

# Fill missing essential columns
for col in ['Labour_Type', 'Location', 'Season', 'Crop_Type', 'Hourly_Rate', 'Rating', 'Skill_Level', 'Name']:
    if col not in DF.columns:
        DF[col] = 'unknown'
    DF[col] = DF[col].fillna('unknown')

print("✅ Data loaded:", DF.shape)

# ---------------------------
# Step 3 - Feature Engineering
# ---------------------------
for col in ['Labour_Type', 'Season', 'Crop_Type', 'Skill_Level', 'Location']:
    DF[col] = DF[col].astype(str).str.strip()

DF['Hourly_Rate'] = pd.to_numeric(DF['Hourly_Rate'], errors='coerce').fillna(0)
DF['Rating'] = pd.to_numeric(DF['Rating'], errors='coerce').fillna(0)

DF['combined_text'] = (
    DF['Labour_Type'] + ' ' + DF['Season'] + ' ' +
    DF['Crop_Type'] + ' ' + DF['Skill_Level'] + ' ' + DF['Location']
)

# Collapse rare labour types
vc = DF['Labour_Type'].value_counts()
rare = vc[vc < 5].index.tolist()
DF['Labour_Type_collapsed'] = DF['Labour_Type'].apply(lambda x: x if x not in rare else 'other')

label_col = 'Labour_Type_collapsed'
print("✅ Feature Engineering Done")

# ---------------------------
# Step 4 - Train/Test Split
# ---------------------------
X = DF['combined_text'].values
y = DF[label_col].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

print(f"🧩 Training Samples: {len(X_train_text)}, Test Samples: {len(X_test_text)}")

# ---------------------------
# Step 5 - TF-IDF Vectorizer
# ---------------------------
vectorizer = TfidfVectorizer(
    stop_words='english', ngram_range=(1,3),
    max_df=0.95, min_df=2, max_features=30000
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("✅ TF-IDF Complete:", X_train.shape)

# ---------------------------
# Step 6 - Model Comparison
# ---------------------------
models = {
    'RandomForest': RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, eval_metric='mlogloss', n_jobs=-1, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=400, learning_rate=0.1, random_state=42)
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, clf in models.items():
    print(f"⚙️ Evaluating {name} ...")
    scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
    results.append({'model': name, 'mean_acc': scores.mean(), 'std_acc': scores.std()})

res_df = pd.DataFrame(results).sort_values('mean_acc', ascending=False)
print("\n📊 Model Comparison:\n", res_df)

plt.figure(figsize=(7,5))
sns.barplot(data=res_df, x='model', y='mean_acc')
plt.title('Model Accuracy Comparison (5-Fold CV)')
plt.ylabel('Mean Accuracy')
plt.savefig('outputs/plots/model_comparison.png')
plt.close()

# ---------------------------
# Step 7 - Train Best Model
# ---------------------------
best_name = res_df.iloc[0]['model']
best_clf = models[best_name]
best_clf.fit(X_train, y_train)
print(f"🏆 Best Model Trained: {best_name}")

joblib.dump({'vectorizer': vectorizer, 'label_encoder': le, 'model': best_clf},
            f'models/best_model_{best_name}.joblib')

# ---------------------------
# Step 8 - Evaluate Model
# ---------------------------
y_pred = best_clf.predict(X_test)
print("✅ Test Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.title(f'Confusion Matrix - {best_name}')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.savefig('outputs/plots/confusion_matrix.png')
plt.close()

# ---------------------------
# Step 9 - Semantic Model + Save Everything
# ---------------------------
print("🔁 Computing semantic embeddings...")
model_semantic = SentenceTransformer('all-MiniLM-L6-v2')

DF['text_for_embed'] = DF['combined_text'] + ' ' + DF['Labour_Type']
DF['embedding'] = DF['text_for_embed'].apply(lambda x: model_semantic.encode(x, convert_to_tensor=True))

# Convert embeddings to numpy
DF_save = DF.copy()
DF_save['embedding'] = DF_save['embedding'].apply(lambda x: x.cpu().numpy())

all_models = {
    'classifier_model': best_clf,
    'vectorizer': vectorizer,
    'label_encoder': le,
    'DF': DF_save,
    'embedding_model_name': 'all-MiniLM-L6-v2'
}

joblib.dump(all_models, 'models/complete_recommendation_model.joblib')
print("✅ Model Saved joblib Successfully")
