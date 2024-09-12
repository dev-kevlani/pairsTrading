import yfinance as yf
import multiprocessing as mp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
import numpy as np
from statsmodels.regression.rolling import RollingOLS
from arch import arch_model
from arch.__future__ import reindexing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

pd.set_option('display.max_columns', None)

def download_stock_data(args):
    ticker, start, end = args
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        return (ticker, data)
    except Exception as e:
        return (ticker, None)

def download_sp500_data_in_parallel(tickers, start, end):
    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.map(download_stock_data, [(ticker, start, end) for ticker in tickers])
    sp500_data = {ticker: data for ticker, data in results if data is not None}
    return sp500_data

def get_indexes_data(indexes, start, end):
    combined_df = pd.DataFrame()
    for index, ticker in indexes.items():
        data = yf.download(index, start=start, end=end)
        data.rename(columns={'Adj Close': ticker}, inplace=True)
        data = data[[ticker]]
        data[f'{ticker}_pctChange'] = data[ticker].pct_change()
        if ticker == 'SP500' or ticker == 'VIX':
            data[f'{ticker}_label'] = data[f'{ticker}_pctChange'].shift(-1)
        combined_df = pd.concat([combined_df, data], axis=1)
    return combined_df

def join_index_stocks(index_df, sp500):
    for ticker in sp500.keys():
        df = sp500[ticker]
        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df = df[[ticker]]
        if df[ticker].isna().sum() > 0.5*len(df):
            continue
        df[f'{ticker}_pctChange'] = df[ticker].pct_change()
        index_df = pd.concat([index_df, df], axis=1)
    index_df.dropna(axis=1, thresh=0.7*len(index_df), inplace=True)
    index_df.dropna(inplace=True)

    return index_df

def get_stats_df(combined_df):

    label = combined_df.filter(like='label')*100
    returns = combined_df.filter(like='pctChange')*100

    stocks_label = pd.concat([label, returns], axis=1)

    stats = pd.DataFrame({
        'count': stocks_label.count(),
        'min': stocks_label.min(),
        'max': stocks_label.max(),
        'mean': stocks_label.mean(),
        'std': stocks_label.std(),
        'median': stocks_label.median(),
        'skew': stocks_label.skew(),
        'kurt': stocks_label.kurt(),
        '25%': stocks_label.quantile(0.25),
        '50%': stocks_label.quantile(.50),
        '75%': stocks_label.quantile(0.75)
    })
    return combined_df, stocks_label, stats.T

def get_correlations_with_p_value(stocks_label):
    corr_df = pd.DataFrame(columns = ['sp_corr', 'spcorr_p', 'vcorr', 'vcorr_p'])

    for col in stocks_label:
        if col != 'SP500_label' and col != 'VIX_label':
            sp,sp_p = stats.pearsonr(stocks_label[col], stocks_label['SP500_label'])
            v,vp = stats.pearsonr(stocks_label[col], stocks_label['VIX_label'])
            corr_df.loc[col] = [round(sp,3), round(sp_p, 3), round(v, 3), round(vp, 3)]
    return corr_df

def get_cointegrated_df():
    # tickers = stocks_label.columns
    # pairs = list(itertools.combinations(tickers,2))
    # cointegrated_df = pd.DataFrame(columns=['coint_val', 'coint_p'])

    # for stock1, stock2 in pairs:
    #     coint_val, coint_p, _ = coint(stocks_label[stock1], stocks_label[stock2])
    #     if coint_p < 0.00000005:
    #         cointegrated_df.loc[f'{stock1}_{stock2}'] = [round(coint_val, 3), round(coint_p, 3)]

    cointegrated_df = pd.read_csv("/Users/admin/Downloads/Fama French/Cointegrated_df.csv")
    cointegrated_df.rename(columns={'Unnamed: 0': 'Cointegration'}, inplace=True)
    cointegrated_df.set_index('Cointegration', inplace=True)
    return cointegrated_df

def plot_pair_spread(combined_df, ticker1, ticker2):
    series_1 = combined_df[ticker1]
    series_2 = combined_df[ticker2]

    series_1_with_const = sm.add_constant(series_1)
    results = sm.OLS(series_2, series_1_with_const).fit()
    b = results.params[ticker1]
    spread = series_2 - b*series_1
    print(spread)
    # plt.figure(figsize=(8,6))
    sns.scatterplot(spread)
    plt.axhline(spread.mean(), color='black')
    plt.title(f'Spread between {ticker1} and {ticker2}')
    plt.legend(['Spread'])
    plt.show()

def plot_pair_spread(combined_df, ticker1, ticker2):
    series_1 = combined_df[ticker1]
    series_2 = combined_df[ticker2]

    series_1_with_const = sm.add_constant(series_1)
    results = sm.OLS(series_2, series_1_with_const).fit()
    b = results.params[ticker1]
    spread = series_2 - b*series_1
    print(spread)
    # plt.figure(figsize=(8,6))
    sns.scatterplot(spread)
    plt.axhline(spread.mean(), color='black')
    plt.title(f'Spread between {ticker1} and {ticker2}')
    plt.legend(['Spread'])
    plt.show()

    return spread

def plot_zscore(series):
    zscore = (series - series.mean()) / np.std(series)
    zscore.plot()
    plt.axhline(zscore.mean(), color='black')
    plt.axhline(2.0, color='red', linestyle='--')
    plt.axhline(-2.0, color='green', linestyle='--')
    plt.legend(['Spread z-score', 'Mean', '+1', '-1'])
    return zscore

def trade_pairs(zscore, spread):
    trades = pd.concat([zscore, spread], axis=1)
    trades.columns = ["signal", "position"]

    trades["side"] = 0.0
    trades.loc[trades.signal <= -2, "side"] = 1
    trades.loc[trades.signal >= 2, "side"] = -1

    return trades

def plot_returns(trades):
    returns = trades.position.pct_change() * trades.side
    returns.cumsum().plot()

    return returns


def main():
    start = '2010-01-01'
    end = '2023-12-31'
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    tickers = sp500_df['Symbol'].to_list()
    indexes = {'^GSPC': 'SP500', '^DJI': 'DJI', '^IXIC': 'NASDAQ', '^RUT': 'Russel2000', '^VIX': 'VIX', 'DX-Y.NYB': 'USD'}

    sp500_data = download_sp500_data_in_parallel(tickers, start, end)
    index_df = get_indexes_data(indexes, start, end)
    combined_df = join_index_stocks(index_df,sp500_data)
    combined_df, stocks_label_df, stats_df = get_stats_df(combined_df)
    corr_df = get_correlations_with_p_value(stocks_label_df)
    cointegrated_df = get_cointegrated_df()
    cointegrated_df[cointegrated_df['coint_val'] <= -30].sort_values(by='coint_val')
    
    return combined_df, stocks_label_df, stats_df, corr_df, cointegrated_df

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method('fork')  # Use 'fork' on macOS
    combined_df, stocks_label_df, stats_df, corr_df, cointegrated_df = main()
    spread = plot_pair_spread(combined_df, 'EL', 'TPR')
    zscore = plot_zscore(spread)
    trades = trade_pairs(zscore, spread)
    returns = plot_returns(trades)
