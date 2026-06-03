import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_transactions(self) -> pd.DataFrame:
        logger.info("Loading transactions...")
        file_path = self.data_dir / "transactions_train.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Optimization: parsing dates and selecting relevant columns
        df = pd.read_csv(
            file_path, 
            usecols=['t_dat', 'customer_id', 'article_id', 'price'],
            parse_dates=['t_dat']
        )
        return df

    def load_articles(self) -> pd.DataFrame:
        logger.info("Loading articles metadata...")
        file_path = self.data_dir / "articles.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        return df
        
    def load_customers(self) -> pd.DataFrame:
        logger.info("Loading customers metadata...")
        file_path = self.data_dir / "customers.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        return df
