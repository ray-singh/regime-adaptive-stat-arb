"""
Stock Universe Definitions for Regime-Adaptive Statistical Arbitrage Platform

Contains predefined lists of liquid US equities for trading strategies.
"""

# Top 200 most liquid US equities (by average daily volume)
# This list represents highly liquid large and mid-cap stocks across sectors
# Updated as of Q4 2025 - adjust periodically based on market conditions

TOP_200_LIQUID_US_EQUITIES = [
    # Mega-cap Technology (20)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL", "ADBE",
    "CRM", "AMD", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC",
    
    # Large-cap Technology & Communication (20)
    "NFLX", "PYPL", "SNOW", "NOW", "PANW", "CRWD", "ZS", "DDOG", "NET", "TEAM",
    "UBER", "ABNB", "LYFT", "DASH", "COIN", "SQ", "SHOP", "MELI", "SE", "BABA",
    
    # Financial Services (20)
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "SPGI",
    "CME", "ICE", "BX", "KKR", "V", "MA", "FIS", "FISV", "ADP", "TFC",
    
    # Healthcare & Biotechnology (20)
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA", "ISRG", "SYK", "BSX", "MDT",
    
    # Consumer Discretionary (20)
    "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "BKNG", "MAR", "GM", "F",
    "RIVN", "LCID", "CMG", "YUM", "DRI", "ULTA", "LULU", "ROST", "HLT", "DG",
    
    # Consumer Staples (20)
    "PG", "KO", "PEP", "WMT", "COST", "PM", "MO", "CL", "MDLZ", "KHC",
    "GIS", "K", "HSY", "CAG", "CPB", "MKC", "SJM", "KMB", "CLX", "CHD",
    
    # Energy (20)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL",
    "DVN", "FANG", "PXD", "HES", "MRO", "APA", "BKR", "NOV", "FTI", "WMB",
    
    # Industrials (20)
    "BA", "HON", "UNP", "CAT", "RTX", "LMT", "GE", "DE", "MMM", "UPS",
    "FDX", "NSC", "CSX", "EMR", "ETN", "ITW", "PH", "CARR", "PCAR", "ROK",
    
    # Materials & Chemicals (20)
    "LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "DOW", "ALB", "CE",
    "PPG", "NUE", "VMC", "MLM", "CF", "MOS", "IFF", "FMC", "EMN", "IP",
    
    # Real Estate & Utilities (20)
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES"
]


# Sector mapping for regime-specific analysis
SECTOR_MAPPING = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AVGO", "ORCL", "ADBE", "CRM", "AMD"],
    "Financials": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA"],
    "Healthcare": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "AMGN"],
    "Consumer": ["AMZN", "HD", "NKE", "MCD", "WMT", "COST", "TGT", "LOW", "SBUX", "PG"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "Industrials": ["BA", "HON", "UNP", "CAT", "RTX", "LMT", "GE", "DE", "UPS", "FDX"],
}


def get_universe(name: str = "top200"):
    """
    Get a predefined stock universe.
    
    Args:
        name: Universe name ('top200', 'sp500', 'custom')
        
    Returns:
        List of ticker symbols
    """
    if name == "top200":
        return TOP_200_LIQUID_US_EQUITIES
    else:
        raise ValueError(f"Unknown universe: {name}")


def get_sector_tickers(sector: str):
    """Get tickers for a specific sector."""
    if sector not in SECTOR_MAPPING:
        raise ValueError(f"Unknown sector: {sector}. Available: {list(SECTOR_MAPPING.keys())}")
    return SECTOR_MAPPING[sector]
