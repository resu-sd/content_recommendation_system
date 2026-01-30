# recommendation.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


csv = "csv_path/sample-data.csv"
df = pd.read_csv(csv)
print(f"Loaded {len(df)} products")


# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r'<[^<]+?>', ' ', str(text))
    text = re.sub(r'\d+\s?(oz|g|ml|cm)?', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()


df['clean_desc'] = df['description'].apply(clean_text)
print("Descriptions cleaned")


# ---------------- CREATE TITLE COLUMN ----------------
# Generate a short title from first 6 words of description
df['title'] = df['description'].str.split().str[:6].str.join(' ')


# ---------------- TF-IDF ----------------
tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 3),
    stop_words='english'
)

tfidf_matrix = tfidf.fit_transform(df['clean_desc'])
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# ---------------- COSINE SIMILARITY ----------------
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("Cosine similarity matrix created")


# ---------------- RECOMMEND FUNCTION ----------------
def recommend(item_id, num=10):
    if item_id not in df['id'].values:
        return []

    idx = df[df['id'] == item_id].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_scores = scores[1:num + 1]
    recommendations = [
        (int(df.iloc[i]['id']), float(score))
        for i, score in top_scores
    ]
    return recommendations


# ---------------- SHOW PRODUCT LIST ----------------
print("\nAvailable Products:")
print("-" * 60)

for _, row in df.iterrows():
    print(f"ID: {row['id']} | {row['title']}...")

print("-" * 60)


# ---------------- USER INPUT ----------------
item_id = int(input("Enter product ID to get recommendations: "))
top_n = int(input("How many recommendations? "))


# ---------------- OUTPUT ----------------
recs = recommend(item_id, num=top_n)

if not recs:
    print("Invalid product ID.")
else:
    print(f"\nTop {top_n} recommendations for product {item_id}:")
    for pid, score in recs:
        print(f"Product ID {pid} — Similarity: {score:.3f}")
