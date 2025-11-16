import yfinance as yf
import pandas as pd
import numpy as np

def load_actions(symbol:str, start_date: str, end_date: str):

    df = yf.download(symbol, start=start_date, end=end_date)
    df = pd.DataFrame(df)
    df.columns =  ['close', 'high', 'low', 'open', 'volume']
    df['date'] = df.index
    df.reset_index(drop=True, inplace=True) 
    df['ano'] = df['date'].dt.year
    df['mes'] = df['date'].dt.month
    df['dia'] = df['date'].dt.day
    df.drop(columns=['date'], inplace=True)

    np.random.seed(42)
    n = 500

    # Ordena por tempo (importante para séries temporais)
    df = df.sort_values(by=['ano', 'mes', 'dia']).reset_index(drop=True)

    return df

if __name__ == "__main__":

    df_main = load_actions("AAPL", "2018-01-01", "2024-12-31")
    print(df_main.head())