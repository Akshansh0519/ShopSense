# Google Colab Embedder Script
# Run this in a Google Colab notebook with a T4 GPU to generate the embeddings in 30 seconds!

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

print("1. Loading Data...")
# Make sure you upload 'articles.csv' and 'item_mapping.pkl' to Colab first!
articles_df = pd.read_csv("articles.csv")

with open("item_mapping.pkl", "rb") as f:
    item_mapping = pickle.load(f)
    
num_items = len(item_mapping)

print("2. Preparing Text...")
df = articles_df[articles_df['article_id'].isin(item_mapping.keys())].copy()
df['item_idx'] = df['article_id'].map(item_mapping)

text_cols = ['product_type_name', 'product_group_name', 'colour_group_name', 'department_name', 'detail_desc']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].fillna('')
        
df['combined_text'] = df.apply(
    lambda x: f"{x.get('product_group_name', '')} {x.get('product_type_name', '')} {x.get('department_name', '')} {x.get('colour_group_name', '')} {x.get('detail_desc', '')}",
    axis=1
)
df = df.sort_values('item_idx').reset_index(drop=True)

print("3. Initializing Neural Network (Using GPU if available)...")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

item_embeddings = np.zeros((num_items, encoder.get_sentence_embedding_dimension()))
valid_indices = df['item_idx'].values
valid_texts = df['combined_text'].tolist()

print(f"4. Encoding {len(valid_texts)} items...")
# This will use the T4 GPU and finish in < 30 seconds
embeddings = encoder.encode(valid_texts, show_progress_bar=True)

item_embeddings[valid_indices] = embeddings

print("5. Saving embeddings...")
np.save("item_embeddings.npy", item_embeddings)
print("Done! Download 'item_embeddings.npy' and place it in your local 'artifacts/' folder.")
