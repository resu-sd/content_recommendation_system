# recommendation.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


csv = "csv_path/sample-data.csv" 

df = pd.read_csv(csv)
print(f"Loaded {len(df)} products")


def clean_text(text):
   
    text = re.sub(r'<[^<]+?>', ' ', text)  
    text = re.sub(r'\d+\s?(oz|g|ml|cm)?', ' ', text)  
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text.lower().strip()

df['clean_desc'] = df['description'].apply(clean_text)
print("Descriptions cleaned")



tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['clean_desc'])
feature_names = tfidf.get_feature_names_out()
print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")


# Compute cosine similarity

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print("Cosine similarity matrix created")


def recommend(item_id, num=10):
    # Check if item exists
    if item_id not in df['id'].values:
        return f"Item {item_id} not found"
    
    # Get index of item
    idx = df[df['id'] == item_id].index[0]

    # Get similarity scores
    scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity (descending)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip first one (itself) and take top-N
    top_scores = scores[1:num+1]

    # Prepare output
    recommendations = [(int(df.iloc[i]['id']), float(score)) for i, score in top_scores]
    return recommendations




item_id = int(input("Enter product ID to get recommendations: "))
top_n = int(input("How many recommendations? "))
    
recs = recommend(item_id, num=top_n)
print(f"\nTop {top_n} recommendations for product {item_id}:")
for pid, score in recs:
     print(f"Product ID {pid} — Similarity: {score:.3f}")
