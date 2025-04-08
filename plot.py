import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


def plot_rsi(data, ticker):
    window = 14
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    plt.figure(figsize=(10, 5))
    plt.plot(data.index, rsi, label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'RSI of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.show()


def plot_bollinger_bands(data, ticker):
    window = 20

    # Fix: Handle multi-index DataFrame just in case
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data[('Close', ticker)]
            data = data.to_frame(name='Close')  # Make it a DataFrame again with single column
        except KeyError:
            print(f"⚠️ Couldn't find Close prices for {ticker}")
            return
    else:
        # Single ticker, normal case
        data = data[['Close']]  # Ensure it's a single-column DataFrame

    # Compute Bollinger Bands
    data['Middle Band'] = data['Close'].rolling(window=window).mean()
    data['Upper Band'] = data['Middle Band'] + 2 * data['Close'].rolling(window=window).std()
    data['Lower Band'] = data['Middle Band'] - 2 * data['Close'].rolling(window=window).std()

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label='Closing Price')
    plt.plot(data.index, data['Middle Band'], label='Middle Band', color='blue')
    plt.plot(data.index, data['Upper Band'], label='Upper Band', color='red')
    plt.plot(data.index, data['Lower Band'], label='Lower Band', color='green')

    plt.title(f'Bollinger Bands of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def plot_pe_ratios(data, ticker, eps):
    if eps is None or eps == 0:
        print(f'Warning: EPS for {ticker} is not avaliable or zero.')
        return
    pe_ratio = data['Close'] / eps
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, pe_ratio, label=f'{ticker} PE Ratio')
    plt.title('PE Ratios of Selected Stocks')
    plt.xlabel('Date')
    plt.ylabel('PE Ratio')
    plt.legend()
    plt.show()


def plot_beta(beta):
    plt.figure(figsize=(10, 5))
    plt.bar(beta.keys(), beta.values(), color='blue')
    plt.title('Beta comparison of Selected Stocks')
    plt.xlabel("Ticker")
    plt.ylabel('Beta')
    plt.show()


def plot_macd(data, ticker):
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, macd, label=f'{ticker} MACD')
    plt.plot(data.index, signal, label=f'{ticker} Signal Line')
    plt.title(f"MACD and Signal Line of {ticker}")
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.show()


# Plots a bar chart comparing beta values of the given stock tickers.
def plot_beta_comparison(tickers, start_date, end_date):
    betas = {}

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        try:
            # Retrieve historical data
            data = stock.history(start=start_date, end=end_date)
            # Get the beta value from stock's info
            beta = stock.info.get('beta')

            # Skip to the next ticker if beta is not available
            if beta is None:
                print(f"Warning: Beta for {ticker} is not available.")
                continue

            betas[ticker] = beta  # Store the beta value

        # Handle errors related to missing data
        except KeyError as e:
            print(f"Error retrieving data for {ticker}: {e}")
        # Handle other unexpected errors
        except Exception as e:
            print(f"An error occurred with ticker {ticker}: {e}")

    # Plotting the bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(betas.keys(), betas.values(), color='blue')
    plt.title('Beta Comparison of Selected Stocks')
    plt.xlabel('Ticker')
    plt.ylabel('Beta')
    plt.show()


def plot_macd(data, ticker):
    """
    Plot the Moving Average Convergence Divergence (MACD) for a given stock.

    Parameters:
    data (DataFrame): The stock data.
    ticker (str): The stock ticker symbol.

    Returns:
    None
    """
    # Calculate the 12-day and 26-day EMA
    ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = data['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD
    macd = ema_12 - ema_26

    # Calculate the signal line
    signal = macd.ewm(span=9, adjust=False).mean()

    # Plot MACD and signal line
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, macd, label=f'{ticker} MACD')
    plt.plot(data.index, signal, label=f'{ticker} Signal Line')
    plt.title(f'MACD and Signal Line of {ticker}')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.show()
