import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime


import yfinance as yf

def get_market_summary():
    try:
        # Use multiple real stocks instead of ^GSPC index
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
        data = yf.download(symbols, period="2d", group_by='ticker', threads=True)

        gainers, losers, active = [], [], []

        for symbol in symbols:
            df = data[symbol]
            if df.empty or len(df) < 2:
                continue

            close_today = df['Close'].iloc[-1]
            close_yesterday = df['Close'].iloc[-2]
            change = close_today - close_yesterday
            percent_change = (change / close_yesterday) * 100
            volume = df['Volume'].iloc[-1]

            gainers.append({"symbol": symbol, "change": round(percent_change, 2)})
            losers.append({"symbol": symbol, "change": round(percent_change, 2)})
            active.append({"symbol": symbol, "volume": int(volume)})

        gainers = sorted(gainers, key=lambda x: x['change'], reverse=True)[:5]
        losers = sorted(losers, key=lambda x: x['change'])[:5]
        active = sorted(active, key=lambda x: x['volume'], reverse=True)[:5]

        return {
            "gainers": gainers,
            "losers": losers,
            "active": active
        }

    except Exception as e:
        return {
            "gainers": [],
            "losers": [],
            "active": [],
            "error": str(e)
        }

def get_trending_stocks(type="all"):
    try:
        if type != "all":
            data = yf.download("^GSPC", period="1d")
            if data.empty:
                return []

            if type == "gainers":
                return data.nlargest(5, 'Close').reset_index().to_dict(orient='records')
            elif type == "losers":
                return data.nsmallest(5, 'Close').reset_index().to_dict(orient='records')
            elif type == "active":
                return data.nlargest(5, 'Volume').reset_index().to_dict(orient='records')

        # default "all"
        return {
            "Top Gainers": yf.download("^GSPC", period="1d").nlargest(5, 'Close').reset_index().to_dict(orient='records'),
            "Top Losers": yf.download("^GSPC", period="1d").nsmallest(5, 'Close').reset_index().to_dict(orient='records'),
            "Most Active": yf.download("^GSPC", period="1d").nlargest(5, 'Volume').reset_index().to_dict(orient='records')
        }

    except Exception as e:
        return {
            "error": f"Failed to fetch trending stocks: {str(e)}"
        }


def get_stock_details(symbol, period=None, start_date=None, end_date=None):
    """Returns stock details with technical indicators"""
    try:
        stock = yf.Ticker(symbol)
        
        # Parse dates if provided as strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Fetch historical data
        if start_date or end_date:
            hist = stock.history(start=start_date, end=end_date)
        else:
            period = period or '1mo'
            hist = stock.history(period=period)
        
        if hist.empty:
            return {
                "status": "error",
                "message": f"No data available for {symbol}",
                "symbol": symbol
            }
            
        # Calculate technical indicators
        hist = calculate_technical_indicators(hist)
        
        # Prepare response data
        data = []
        for index, row in hist.iterrows():
            change = round(row['Close'] - row['Open'], 2)
            data.append({
                "Date": index.strftime('%Y-%m-%d'),
                "Open": round(row['Open'], 2),
                "High": round(row['High'], 2),
                "Low": round(row['Low'], 2),
                "Close": round(row['Close'], 2),
                "Volume": int(row['Volume']),
                "Change": change,
                "ChangePercent": round((change / row['Open']) * 100, 2) if row['Open'] != 0 else 0,
                "MA_20": round(row['MA_20'], 2) if 'MA_20' in row and not pd.isna(row['MA_20']) else None,
                "MA_50": round(row['MA_50'], 2) if 'MA_50' in row and not pd.isna(row['MA_50']) else None,
                "RSI": round(row['RSI'], 2) if 'RSI' in row and not pd.isna(row['RSI']) else None,
                "MACD": round(row['MACD'], 2) if 'MACD' in row and not pd.isna(row['MACD']) else None,
                "Signal_Line": round(row['Signal_Line'], 2) if 'Signal_Line' in row and not pd.isna(row['Signal_Line']) else None,
                "Upper_Band": round(row['Upper_Band'], 2) if 'Upper_Band' in row and not pd.isna(row['Upper_Band']) else None,
                "Middle_Band": round(row['Middle_Band'], 2) if 'Middle_Band' in row and not pd.isna(row['Middle_Band']) else None,
                "Lower_Band": round(row['Lower_Band'], 2) if 'Lower_Band' in row and not pd.isna(row['Lower_Band']) else None
            })
        
        return {
            "status": "success",
            "symbol": symbol,
            "data": data,
            "period": period,
            "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
            "end_date": end_date.strftime('%Y-%m-%d') if end_date else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "symbol": symbol
        }

def calculate_technical_indicators(data):
    """Calculates technical indicators for stock data"""
    if len(data) < 20:  # Minimum required data points
        # Initialize empty indicators if insufficient data
        for col in ['MA_20', 'MA_50', 'RSI', 'MACD', 'Signal_Line', 
                   'Upper_Band', 'Middle_Band', 'Lower_Band']:
            data[col] = np.nan
        return data
    
    # Moving Averages
    data['MA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['MA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    
    # RSI Calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD Calculation
    exp12 = data['Close'].ewm(span=12, adjust=False).mean()
    exp26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    data['Middle_Band'] = data['Close'].rolling(window=20).mean()
    data['Upper_Band'] = data['Middle_Band'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_Band'] = data['Middle_Band'] - (data['Close'].rolling(window=20).std() * 2)

    return data

# Add this function to model.py

def get_index_summary():
    try:
        indices = {
            '^GSPC': 'S&P 500',
            '^IXIC': 'NASDAQ',
            '^DJI': 'Dow Jones'
        }
        data = yf.download(list(indices.keys()), period="1d", group_by='ticker', threads=True)
        summary = []

        for symbol, name in indices.items():
            df = data[symbol]
            if df.empty:
                continue
            last_close = df['Close'].iloc[-1]
            prev_close = df['Close'].iloc[-2] if len(df) > 1 else last_close
            change = last_close - prev_close
            percent = (change / prev_close) * 100 if prev_close else 0
            summary.append({
                "name": name,
                "value": round(last_close, 2),
                "change": round(change, 2),
                "percent": round(percent, 2),
                "up": change >= 0
            })
        return summary
    except Exception as e:
        return []
