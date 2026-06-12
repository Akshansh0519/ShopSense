import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_synthetic_data(num_users=1000, num_items=500, num_interactions=10000):
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic customers...")
    customers = pd.DataFrame({
        'customer_id': [f"user_{i:05d}" for i in range(num_users)],
        'age': np.random.randint(18, 65, num_users),
        'postal_code': np.random.choice(['A', 'B', 'C', 'D'], num_users)
    })
    
    print("Generating synthetic articles...")
    categories = ['Outerwear', 'Tops', 'Trousers', 'Accessories', 'Shoes']
    articles = pd.DataFrame({
        'article_id': [f"item_{i:05d}" for i in range(num_items)],
        'product_type_name': np.random.choice(['Sweater', 'T-shirt', 'Jeans', 'Hat', 'Sneaker'], num_items),
        'product_group_name': np.random.choice(categories, num_items),
        'colour_group_name': np.random.choice(['Black', 'White', 'Blue', 'Red', 'Green'], num_items),
        'department_name': np.random.choice(['Men', 'Women', 'Kids'], num_items),
        'detail_desc': ['A very nice synthetic product.'] * num_items
    })
    
    print("Generating synthetic interactions...")
    # Power law distribution for item popularity
    item_probs = np.random.pareto(a=2, size=num_items)
    item_probs /= item_probs.sum()
    
    user_ids = np.random.choice(customers['customer_id'], size=num_interactions)
    item_ids = np.random.choice(articles['article_id'], size=num_interactions, p=item_probs)
    
    # Generate dates over the last 90 days
    end_date = datetime.now()
    dates = [end_date - timedelta(days=np.random.randint(0, 90)) for _ in range(num_interactions)]
    
    transactions = pd.DataFrame({
        't_dat': dates,
        'customer_id': user_ids,
        'article_id': item_ids,
        'price': np.random.uniform(10, 100, num_interactions)
    })
    
    # Sort by date
    transactions = transactions.sort_values('t_dat').reset_index(drop=True)
    
    print("Saving to data/raw/...")
    customers.to_csv(raw_data_dir / "customers.csv", index=False)
    articles.to_csv(raw_data_dir / "articles.csv", index=False)
    transactions.to_csv(raw_data_dir / "transactions_train.csv", index=False)
    
    print(f"Generated {num_users} users, {num_items} items, {num_interactions} transactions.")
    
if __name__ == "__main__":
    np.random.seed(42)
    create_synthetic_data()
