import os
import json
import logging
from pathlib import Path
import pyarrow.parquet as pq

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
COUNT_PATH = DATA_DIR / "count_data.parquet"
STATIC_DIR = PROJECT_ROOT / "static"
OUTPUT_PATH = STATIC_DIR / "genes.json"

def extract_genes():
    """Extract distinct gene names from the parquet file and save to JSON."""
    if not COUNT_PATH.exists():
        logger.error(f"Count data file not found at {COUNT_PATH}")
        return

    logger.info(f"Reading genes from {COUNT_PATH}...")
    
    try:
        parquet_file = pq.ParquetFile(COUNT_PATH)
        genes = set()
        
        # Iterate through row groups to avoid loading entire file at once
        for i in range(parquet_file.num_row_groups):
            table = parquet_file.read_row_group(i, columns=['Gene'])
            chunk_genes = table['Gene'].to_pylist()
            genes.update(g.strip().upper() for g in chunk_genes if g)
            
        sorted_genes = sorted(list(genes))
        logger.info(f"Full gene count: {len(sorted_genes)}")
        
        # Ensure static directory exists
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(sorted_genes, f)
            
        logger.info(f"Successfully saved {len(sorted_genes)} genes to {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to extract genes: {e}")

if __name__ == "__main__":
    extract_genes()
