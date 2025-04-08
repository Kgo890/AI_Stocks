import numpy as np
from datetime import datetime, timedelta
from IPython.display import display, Markdown
import yfinance as yf
from langchain_openai import ChatOpenAI
from scipy.optimize import minimize


# Function to initialize OpenAI LLM model
def initialize_llm(api_key):
    """
    Initialize the ChatOpenAI language model.

    Parameters:
    api_key (str): The API key for accessing the OpenAI service.

    Returns:
    ChatOpenAI: The initialized language model.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    return llm


def calculate_date_range(years):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def get_llm_response(llm, prompt):
    response = llm(prompt)
    return display(Markdown(response.content))


def calculate_kpis(tickers, start_date, end_date):
    """
    Calculate KPIs for a list of stocks over a given time period.

    Parameters:
    tickers (list): A list of stock ticker symbols.
    start_date (str): The start date for the analysis.
    end_date (str): The end date for the analysis.

    Returns:
    dict: A dictionary containing the KPIs for each stock.
    """
    kpi_data = {}
    for ticker in tickers:
        # Download historical stock data
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        kpi_data[ticker] = {}

        # Calculate RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()  # Average losses
        rs = gain / loss  # Relative strength
        rsi = 100 - (100 / (1 + rs))  # RSI formula
        kpi_data[ticker]['RSI'] = rsi

        # Calculate Bollinger Bands
        middle_band = data['Close'].rolling(window=20).mean()  # Middle band (SMA)
        upper_band = middle_band + 2 * data['Close'].rolling(window=20).std()  # Upper band
        lower_band = middle_band - 2 * data['Close'].rolling(window=20).std()  # Lower band
        kpi_data[ticker]['Bollinger Bands'] = {
            'Middle Band': middle_band,
            'Upper Band': upper_band,
            'Lower Band': lower_band
        }

        # Calculate P/E Ratio
        try:
            eps = stock.info.get('trailingEps')  # Get trailing EPS
            if eps and eps != 0:
                pe_ratio = data['Close'] / eps
                kpi_data[ticker]['P/E Ratio'] = pe_ratio  # Calculate P/E ratio
            else:
                kpi_data[ticker]['P/E Ratio'] = None
        except Exception as e:
            kpi_data[ticker]['P/E Ratio'] = None
            print(f"An error occurred with ticker {ticker} P/E Ratio: {e}")

        # Calculate Beta
        try:
            beta = stock.info.get('beta')  # Get beta value
            kpi_data[ticker]['Beta'] = beta
        except Exception as e:
            kpi_data[ticker]['Beta'] = None
            print(f"An error occurred with ticker {ticker} Beta: {e}")

        # Calculate MACD
        ema_12 = data['Close'].ewm(span=12, adjust=False).mean()  # 12-day EMA
        ema_26 = data['Close'].ewm(span=26, adjust=False).mean()  # 26-day EMA
        macd = ema_12 - ema_26  # MACD line
        signal = macd.ewm(span=9, adjust=False).mean()  # Signal line
        kpi_data[ticker]['MACD'] = {
            'MACD': macd,
            'Signal Line': signal
        }

    return kpi_data


# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio performance metrics.

    Parameters:
    weights (array): Asset weights in the portfolio.
    mean_returns (Series): Mean returns for each asset.
    cov_matrix (DataFrame): Covariance matrix of asset returns.

    Returns:
    float: Portfolio returns.
    float: Portfolio standard deviation.
    """
    # Calculate the expected portfolio return
    returns = np.sum(mean_returns * weights)

    # Calculate the portfolio standard deviation (volatility)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std


# Function to calculate the negative Sharpe ratio
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Calculate the negative Sharpe ratio for a given portfolio.

    Parameters:
    weights (array): Asset weights in the portfolio.
    mean_returns (Series): Mean returns for each asset.
    cov_matrix (DataFrame): Covariance matrix of asset returns.
    risk_free_rate (float): Risk-free rate.

    Returns:
    float: Negative Sharpe ratio.
    """
    # Calculate the negative Sharpe ratio of the portfolio
    p_returns, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    # Return the negative Sharpe ratio (to minimize in optimization problems)
    return -(p_returns - risk_free_rate) / p_std


# Function to find the portfolio with the maximum Sharpe ratio
def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """
    Find the portfolio with the maximum Sharpe ratio.

    Parameters:
    mean_returns (Series): Mean returns for each asset.
    cov_matrix (DataFrame): Covariance matrix of asset returns.
    risk_free_rate (float): Risk-free rate.

    Returns:
    OptimizeResult: The optimization result containing the portfolio weights.
    """
    # Number of assets in the portfolio
    num_assets = len(mean_returns)

    # Define the arguments for the optimization function
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Set up constraints (weights must sum to 1)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Define bounds for each asset weight (between 0 and 1)
    bounds = tuple((0, 1) for asset in range(num_assets))

    # Perform optimization to maximize the Sharpe ratio (minimize the negative Sharpe ratio)
    result = minimize(negative_sharpe_ratio, num_assets * [1. / num_assets, ], args=args,
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result
