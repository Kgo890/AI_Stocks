from pypfopt import risk_models, expected_returns, BlackLittermanModel, EfficientFrontier, black_litterman
from dotenv import load_dotenv
import os
from functions import *
from plot import *

load_dotenv()

api_key = os.getenv("api_key")

llm = initialize_llm(api_key=api_key)
years = 2

assets = [
    "Apple (AAPL)",
    "Amazon (AMZN)",
    "Bitcoin (BTC-USD)",
    "Alphabet (GOOGL)",
    "Meta (META)",
    "Microsoft (MSFT)",
    "Nvidia (NVDA)",
    "S&P 500 index (SPY)",
    "Tesla (TSLA)"
    "KULR (KULR)"
]

query = f"""
fetch me the tickers from the following assets: {assets}
Your output must be sorted alhabetically by the ticker and it should be like this:
tickers = ['AAPL', 'AMZN', 'BTC-USD', 'GOOGL', 'META', 'MSFT', 'NVDA', 'SPY', 'TSLA']
"""

get_llm_response(llm=llm, prompt=query)

tickers = ['AAPL', 'AMZN', 'BTC-USD', 'GOOGL', 'KULR', 'META', 'MSFT', 'NVDA', 'SPY', 'TSLA']
# Calculate the start and end dates
start_date, end_date = calculate_date_range(years)

# Fetch historical stock data using yfinance
data = yf.download(tickers, start=start_date, end=end_date)

# Display the first few rows of the fetched data
data.head()

# Download historical data for each ticker and plot its RSI
for ticker in tickers:
    data_ticker = yf.download(ticker, start=start_date, end=end_date)
    plot_rsi(data_ticker, ticker)

for ticker in tickers:
    data_ticker = yf.download(ticker, start=start_date, end=end_date)
    plot_bollinger_bands(data_ticker, ticker)

# Download historical data for each ticker and plot its P/E ratios
for ticker in tickers:
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps')  # Retrieve the trailing EPS value from the stock's info
    data_ticker = yf.download(ticker, start=start_date, end=end_date)
    plot_pe_ratios(data_ticker, ticker, eps)

# Plot beta comparison for a list of tickers over a specified date range
plot_beta_comparison(tickers, start_date, end_date)

# Download historical data for each ticker and plot its MACD
for ticker in tickers:
    data_ticker = yf.download(ticker, start=start_date, end=end_date)
    plot_macd(data_ticker, ticker)

# Calculate KPIs for the defined tickers and date range
kpi_data = calculate_kpis(tickers, start_date, end_date)

# Create a prompt to generate an executive summary with recommendations based on KPI data
prompt = f""" Read this data {kpi_data} and provide an executive summary with recommendations"""

# Get the response from the LLM based on the provided prompt
get_llm_response(llm=llm, prompt=prompt)

# Download full dataset
data = yf.download(tickers, start=start_date, end=end_date)

close_prices = data['Close']

# Now continue with returns, mean, etc.
returns = close_prices.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()
risk_free_rate = 0.04

# Optimize the portfolio for maximum Sharpe ratio
optimal_portfolio = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
optimal_weights = optimal_portfolio.x

# Store the optimal weights in a dictionary and print the result
weights_dict = {tickers[i]: round(optimal_weights[i], 2) for i in range(len(tickers))}
print(weights_dict)

# Define risk-free rate
risk_free_rate = 0.001

# Fetch historical stock data
df = yf.download(tickers, start=start_date, end=end_date)['Close']

# Calculate the sample mean returns and the covariance matrix
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)

## Define market capitalizations
mcap = {}
# Iterate over each ticker symbol to retrieve market capitalization
for ticker in tickers:
    stock = yf.Ticker(ticker)
    try:
        # Attempt to get the market capitalization from stock info
        mcap[ticker] = stock.info['marketCap']
    except KeyError:
        # If the market capitalization is not available, set it to None
        mcap[ticker] = None

# Manually set the market capitalization for the S&P 500 index (SPY)
mcap['SPY'] = 45000000000000

# Define beliefs (Microsoft will outperform Google by 5%)
Q = np.array([0.05])  # Define the vector of expected returns differences (our belief)
P = np.zeros((1, len(tickers)))  # Initialize the matrix of constraints
P[0, tickers.index('MSFT')] = 1  # Set the coefficient for Microsoft to 1
P[0, tickers.index('GOOGL')] = -1  # Set the coefficient for Google to -1

# Calculate the market implied returns
market_prices = df["SPY"]
delta = black_litterman.market_implied_risk_aversion(market_prices)
market_prior = black_litterman.market_implied_prior_returns(mcap, delta, S, risk_free_rate)

# Create the Black-Litterman model
bl = BlackLittermanModel(S,  # Covariance matrix of asset returns
                         Q=Q,  # Vector of expected returns differences (our beliefs)
                         P=P,  # Matrix representing the assets involved in the beliefs
                         pi=market_prior,  # Equilibrium market returns
                         market_weights=market_prior,
                         # Market capitalization weights (used for the equilibrium returns)
                         risk_free_rate=risk_free_rate)  # Risk-free rate for the model

# Get the adjusted returns and covariance matrix
bl_returns = bl.bl_returns()
bl_cov = bl.bl_cov()

# Optimize the portfolio for maximum Sharpe ratio
ef = EfficientFrontier(bl_returns,
                       bl_cov)  # Create an Efficient Frontier object with the adjusted returns and covariance matrix
weights = ef.max_sharpe(
    risk_free_rate=risk_free_rate)  # Calculate the optimal portfolio weights that maximize the Sharpe ratio, considering the risk-free rate
cleaned_weights = ef.clean_weights()  # Clean up the weights to remove very small values for better interpretability

# Print the optimal weights and portfolio performance
print(cleaned_weights)
ef.portfolio_performance(verbose=True)
