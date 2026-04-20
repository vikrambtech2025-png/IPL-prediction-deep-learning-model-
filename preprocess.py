import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(path="data/ipl.csv"):

    df = pd.read_csv(path, low_memory=False)

    # Runs column
    if 'runs_total' in df.columns:
        runs_col = 'runs_total'
    else:
        runs_col = 'batter_runs'

    # Wicket detection
    df['wickets'] = df['wicket_kind'].notna().astype(int)

    # Over precision
    df['over'] = df['over'] + df['ball'] / 10

    # Sort (VERY IMPORTANT)
    df = df.sort_values(by=['match_id', 'innings', 'over'])

    # Cumulative score per innings
    df['total_score'] = df.groupby(['match_id', 'innings'])[runs_col].cumsum()

    df = df[['over', runs_col, 'wickets', 'total_score']]
    df.columns = ['over', 'runs', 'wickets', 'total_score']

    return df


def scale_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler


def create_sequences(data, seq_len=10):

    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][-1])

    return np.array(X), np.array(y)