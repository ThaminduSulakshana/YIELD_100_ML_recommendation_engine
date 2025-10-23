from flask import Flask, request, jsonify, render_template
import joblib, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# -----------------------------
# Load model and components
# -----------------------------
model_data = joblib.load('models/complete_recommendation_model.pkl')

clf = model_data['classifier_model']
vectorizer = model_data['vectorizer']
le = model_data['label_encoder']
DF = model_data['DF']
semantic_model = SentenceTransformer(model_data['embedding_model_name'])

print("✅ Models Loaded Successfully")

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    """Render the frontend HTML page."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle recommendation queries from frontend."""
    data = request.get_json()
    query = data.get('query', '')
    top_k = int(data.get('top_k', 5))

    if not query.strip():
        return jsonify({'error': 'Empty query'}), 400

    # ---- Predict category ----
    qv = vectorizer.transform([query])
    pred = clf.predict(qv)[0]
    pred_label = le.inverse_transform([pred])[0]

    # ---- Compute embedding ----
    query_emb = semantic_model.encode(query, convert_to_tensor=True)

    # ---- Filter DF by predicted label ----
    filtered = DF[DF['Labour_Type_collapsed'] == pred_label].copy()

    # ---- Compute similarity ----
    filtered['similarity'] = filtered['embedding'].apply(
        lambda x: util.cos_sim(query_emb, x).item()
    )

    # ---- Optional: rate filter (e.g., "under 1000") ----
    m = re.search(r'(under|below|less than|more than|above)\s*(\d+)', query.lower())
    if m:
        direction, val = m.groups()
        val = float(val)
        if direction in ['under', 'below', 'less than']:
            filtered = filtered[filtered['Hourly_Rate'] <= val]
        else:
            filtered = filtered[filtered['Hourly_Rate'] >= val]

    # ---- Combine similarity + rating ----
    filtered['score'] = 0.7 * filtered['similarity'] + 0.3 * (filtered['Rating'] / 5.0)
    top_matches = filtered.sort_values(by='score', ascending=False).head(top_k)

    # ---- Prepare JSON results ----
    results = top_matches[
        ['Name', 'Location', 'Labour_Type', 'Season', 'Crop_Type', 'Hourly_Rate', 'Rating', 'score']
    ].to_dict(orient='records')

    return jsonify({
        'query': query,
        'predicted_labour_type': pred_label,
        'recommendations': results
    })

# -----------------------------
# Run server
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
