from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from functools import wraps
from model import get_market_summary, get_trending_stocks
from model import get_stock_details
from model import get_index_summary

app = Flask(__name__, static_folder='.', template_folder='templates')
app.secret_key = "your-secret-key"  # Replace with a real secret in production
app.permanent_session_lifetime = timedelta(days=1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        if username:
            session['user'] = username
            return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/')
def index():
    market_summary = get_market_summary()
    trending_stocks = get_trending_stocks()
    index_summary = get_index_summary()
    return render_template("index.html", 
        market_summary=market_summary,
        trending_stocks=trending_stocks,
        index_summary=index_summary,
        user=session.get('user')
    )


@app.route("/stock/<symbol>")
def stock_details(symbol):
    period = request.args.get("period", "1mo")
    data = get_stock_details(symbol, period)

    if isinstance(data, dict) and 'error' in data:
        return render_template("stock_details.html", symbol=symbol.upper(), stock_data=[])

    return render_template("stock_details.html", symbol=symbol.upper(), stock_data=data)

def handle_errors(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred"}), 500
    return wrapper

@app.route('/predict', methods=['POST'])
@handle_errors
def predict():
    symbol = request.form.get('stockSymbol', '').strip().upper()
    logger.info(f"Prediction request for symbol: {symbol}")
    
    if not symbol:
        return jsonify({"error": "No stock symbol provided"}), 400

    # Validate symbol format
    if not symbol.isalpha() or len(symbol) > 5:
        return jsonify({"error": "Invalid stock symbol format"}), 400

    try:
        # Try with 1 month data first
        data = yf.download(symbol, period='1mo', progress=False, timeout=10)
        
        if data.empty:
            # Fallback to max period if no recent data
            data = yf.download(symbol, period='max', progress=False, timeout=10)
            if data.empty:
                return jsonify({"error": f"No historical data available for {symbol}"}), 404

        # Ensure we have enough data points
        if len(data) < 5:
            return jsonify({"error": f"Insufficient data points for {symbol}"}), 400

        # Get prediction from model (replace with your actual prediction logic)
        last_close = data['Close'].iloc[-1]
        
        # Simple prediction example - replace with your actual model
        prediction = last_close * 1.05  # 5% increase as example
        
        return jsonify({
            "symbol": symbol,
            "prediction": float(prediction),
            "last_close": float(last_close),
            "currency": "USD",
            "last_updated": datetime.now().strftime('%Y-%m-%d')
        })

    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {str(e)}")
        return jsonify({
            "error": f"Failed to generate prediction for {symbol}",
            "details": str(e)
        }), 500

@app.route('/api/historical', methods=['GET'])
@handle_errors
def historical_data():
    symbol = request.args.get('symbol', '').strip().upper()
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    interval = request.args.get('interval', '1d')  # 1d, 1wk, 1mo
    
    # Validate inputs
    if not symbol:
        return jsonify({"error": "Stock symbol required"}), 400
    
    # Set default date range (last 3 months)
    if not start_date or not end_date:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
    else:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        if start_date > end_date:
            return jsonify({"error": "Start date must be before end date"}), 400

    try:
        # Fetch data with multiple fallbacks
        data = None
        for period in ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']:
            try:
                data = yf.download(
                    symbol,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=interval,
                    progress=False,
                    timeout=10
                )
                if not data.empty:
                    break
            except:
                continue
        
        if data is None or data.empty:
            return jsonify({"error": f"No data available for {symbol} in this date range"}), 404
        
        # Process data
        historical = []
        for date, row in data.iterrows():
            historical.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
                "volume": int(row['Volume']),
                "adj_close": float(row['Adj Close']) if 'Adj Close' in row else None
            })
        
        return jsonify({
            "symbol": symbol,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "interval": interval,
            "data": historical
        })

    except Exception as e:
        logger.error(f"Historical data failed for {symbol}: {str(e)}")
        return jsonify({
            "error": f"Failed to fetch historical data for {symbol}",
            "details": str(e)
        }), 500
    
@app.route('/api/trending')
def api_trending():
    """Main trending stocks endpoint"""
    return jsonify(get_trending_stocks())  # This uses the default "all" type

@app.route('/api/trending/gainers')
def top_gainers():
    return jsonify(get_trending_stocks(type="gainers"))

@app.route('/api/trending/losers')
def top_losers():
    return jsonify(get_trending_stocks(type="losers"))

@app.route('/api/trending/active')
def most_active():
    return jsonify(get_trending_stocks(type="active"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)