import pandas as pd
import json
from pathlib import Path

class SimpleTickLoader:
    def __init__(self, symbols_path: str, data_folder: str):
        """
        Simple tick data loader
        
        Args:
            symbols_path: Path to JSON file with symbols list
            data_folder: Path to folder containing symbol subfolders
        """
        self.symbols_path = Path(symbols_path)
        self.data_folder = Path(data_folder)
        self.symbols = self.load_symbols()
    
    def load_symbols(self) -> list:
        """Load symbols from JSON file"""
        with open(self.symbols_path, 'r') as f:
            return json.load(f)
    
    def load_one(self, symbol: str) -> pd.DataFrame:
        """Load data for one symbol"""
        folder = self.data_folder / symbol
        
        if not folder.exists():
            print(f"Folder not found: {symbol}")
            return pd.DataFrame()
        
        # Get all parquet files
        files = list(folder.glob("*.parquet"))
        
        if not files:
            print(f"No parquet files in: {symbol}")
            return pd.DataFrame()
        
        # Load all files
        data = []
        for file in files:
            df = pd.read_parquet(file)
            if not df.empty:
                data.append(df)
        
        if not data:
            return pd.DataFrame()
        
        # Combine and process
        df = pd.concat(data, ignore_index=True)
        df['DateTime'] = pd.to_datetime(df['DateTime']) + pd.Timedelta(hours=2)
        df['Symbol'] = symbol
        df = df.sort_values('DateTime')
        
        return df
    
    def load_some(self, symbols: list) -> pd.DataFrame:
        """Load data for multiple symbols"""
        all_data = []
        
        for symbol in symbols:
            print(f"Loading {symbol}...")
            df = self.load_one(symbol)
            if not df.empty:
                all_data.append(df)
                print(f"  ✓ {len(df)} ticks")
            else:
                print(f"  ✗ Failed")
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def load_all(self) -> pd.DataFrame:
        """Load data for all symbols"""
        return self.load_some(self.symbols)
    
    def check_symbols(self) -> dict:
        """Check which symbols have data"""
        result = {"available": [], "missing": []}
        
        for symbol in self.symbols:
            folder = self.data_folder / symbol
            files = list(folder.glob("*.parquet")) if folder.exists() else []
            
            if files:
                result["available"].append({
                    "symbol": symbol,
                    "files": len(files)
                })
            else:
                result["missing"].append(symbol)
        
        return result


# Quick load functions
def load_symbol(symbol: str) -> pd.DataFrame:
    """Load one symbol"""
    loader = SimpleTickLoader(
        r"your\own\path\ML Model\Model Parameters\symbol_list.json",
        r"path\to\your\final\dataset"
    )
    return loader.load_one(symbol)

def load_symbols(symbols: list) -> pd.DataFrame:
    """Load multiple symbols"""
    loader = SimpleTickLoader(
        r"your\own\path\ML Model\Model Parameters\symbol_list.json",
        r"path\to\your\final\dataset"
    )
    return loader.load_some(symbols)

def load_all_symbols() -> pd.DataFrame:
    """Load all symbols"""
    loader = SimpleTickLoader(
        r"your\own\path\ML Model\Model Parameters\symbol_list.json",
        r"path\to\your\final\dataset"
    )
    return loader.load_all()