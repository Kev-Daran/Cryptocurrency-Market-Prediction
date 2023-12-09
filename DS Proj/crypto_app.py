import streamlit as st
from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import requests

# Function to get historical cryptocurrency data using CoinGecko API
def get_crypto_data(symbol, days=90):
    cg = CoinGeckoAPI()
    crypto_data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=days)

    data_df = pd.DataFrame(crypto_data['prices'], columns=['timestamp', 'price'])
    data_df['Date'] = pd.to_datetime(data_df['timestamp'], unit='ms')
    candlestick_data = data_df.groupby(data_df['Date'].dt.date).aggregate({'price': {'max', 'min', 'first', 'last'}})
    candlestick_data.columns = ['closing_price', 'max_price', 'min_price', 'opening_price']
    candlestick_data.reset_index(inplace=True)  # Reset index to include 'Date'
    return data_df, candlestick_data

# Function to get Fear and Greed Index data
def get_fng_data():
    url = 'https://api.alternative.me/fng/?limit=91&date_format=cn'
    r = requests.get(url)
    data = r.json()
    temp_df = pd.DataFrame(data['data'])
    temp_df['value'] = temp_df['value'].astype(int)
    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
    temp_df.index = temp_df['timestamp']

    # Extract value_classification based on your criteria
    temp_df['value_classification'] = np.where(temp_df['value'] > 60, 'Greedy',
                                                np.where(temp_df['value'] < 40, 'Scared', 'Neutral'))

    return temp_df[['value', 'value_classification', 'timestamp']]

# Function to train a logistic regression model
def train_logistic_regression_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Function to predict price change for a given time duration
def predict_price_change(model, features):
    return model.predict(features)

# Streamlit web app
def main():
    st.title('Cryptocurrency Analysis App')

    # Choose cryptocurrencies to analyze
    selected_crypto = st.selectbox('Select Cryptocurrency', ['bitcoin', 'ethereum', 'litecoin', 'ripple', 'cardano'])

    # Get current price and refresh button
    current_data, _ = get_crypto_data(selected_crypto, days=1)
    current_price = current_data.iloc[-1]['price']
    current_time = current_data.iloc[-1]['Date']

    if st.button('Check'):
        current_data, _ = get_crypto_data(selected_crypto, days=1)
        current_price = current_data.iloc[-1]['price']
        current_time = current_data.iloc[-1]['Date']

    st.write(f'Current Price of {selected_crypto.capitalize()} as of {current_time}  UTC -----> {current_price:.2f} USD')

    # Display past 5 days data in a table
    _, past_5_days_data = get_crypto_data(selected_crypto, days=5)
    st.subheader('Past 5 Days Data')
    st.table(past_5_days_data[['Date', 'max_price', 'min_price', 'opening_price', 'closing_price']].reset_index(drop=True))

    # Display last 90 days data in a figure/boxplot
    _, historical_data = get_crypto_data(selected_crypto, days=90)
    st.subheader('Last 90 Days Data')
    fig = go.Figure(data=[go.Candlestick(x=historical_data['Date'],
                                     open=historical_data['opening_price'],
                                     low=historical_data['min_price'],
                                     high=historical_data['max_price'],
                                     close=historical_data['closing_price'])])

    fig.update_layout(title=f'{selected_crypto.capitalize()} Prices Over the Last 90 Days',
                  yaxis_title='Price (USD)',
                  xaxis_title='Date')

    st.plotly_chart(fig)

    # Logistic Regression for predicting increase/decrease
    _, historical_data = get_crypto_data(selected_crypto, days=90)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(historical_data[['max_price', 'min_price', 'opening_price', 'closing_price']])
    features = scaled_data[:-1]  # Exclude the last day for prediction
    target = (historical_data['closing_price'].shift(-1) > historical_data['closing_price']).iloc[:-1]

    model = train_logistic_regression_model(features, target)

    # Display prediction buttons
    st.subheader('Predict Price Change for Different Durations')
    if st.button('Predict Next Day'):
        prediction_duration = 'Next Day'
        prediction_features = scaled_data[-1].reshape(1, -1)
        prediction_result = predict_price_change(model, prediction_features)
        display_prediction_result(prediction_result, prediction_duration)

    if st.button('Predict 1 Month'):
        prediction_duration = '1 Month'
        prediction_features = scaled_data[-30:].mean(axis=0).reshape(1, -1)
        prediction_result = predict_price_change(model, prediction_features)
        display_prediction_result(prediction_result, prediction_duration)

    if st.button('Predict 1 Year'):
        prediction_duration = '1 Year'
        prediction_features = scaled_data[-365:].mean(axis=0).reshape(1, -1)
        prediction_result = predict_price_change(model, prediction_features)
        display_prediction_result(prediction_result, prediction_duration)

    if st.button('Predict 3 Years'):
        prediction_duration = '3 Years'
        prediction_features = scaled_data[-365 * 3:].mean(axis=0).reshape(1, -1)
        prediction_result = predict_price_change(model, prediction_features)
        display_prediction_result(prediction_result, prediction_duration)

    # Display past 5 days sentiment values in a table
    fng_data = get_fng_data()

    # Get today's date
    today_date = pd.to_datetime('today').date()

    # Filter data for the past five days, including today
    past_5_days_data = fng_data[fng_data.index.date >= (today_date - pd.DateOffset(4)).date()]

    st.subheader('Past 5 Days Sentiment Values')
    st.table(past_5_days_data[['timestamp', 'value', 'value_classification']])

    # Choose the time duration for historical data
    duration_options = ['7 days', '1 month', '3 months', '1 year']
    selected_duration = st.selectbox('Select Time Duration for Historical Data', duration_options)

    # Plot historical data based on selected duration
    st.subheader(f'Historical Data for the past {selected_duration} Based on Sentiment Values')
    
    if selected_duration == '7 days':
        plot_data = fng_data.loc[fng_data.index >= fng_data.index.max() - pd.DateOffset(7)]
    elif selected_duration == '1 month':
        plot_data = fng_data.loc[fng_data.index >= fng_data.index.max() - pd.DateOffset(30)]
    elif selected_duration == '3 months':
        plot_data = fng_data.loc[fng_data.index >= fng_data.index.max() - pd.DateOffset(90)]
    else:  # 1 year
        plot_data = fng_data.loc[fng_data.index >= fng_data.index.max() - pd.DateOffset(365)]

    st.plotly_chart(go.Figure(data=[go.Scatter(x=plot_data['timestamp'], y=plot_data['value'],
                                               mode='lines+markers', name='Sentiment Value')],
                              layout=go.Layout(title=f'{selected_crypto.capitalize()} Sentiment Values Over the Last {selected_duration}',
                                              xaxis_title='Date', yaxis_title='Sentiment Value')))

def display_prediction_result(prediction_result, duration):
    st.subheader(f'Prediction for {duration}:')
    if prediction_result[-1]:
        st.subheader(f'**The value is predicted to increase.**')
    else:
        st.subheader(f'**The value is predicted to decrease.**')

if __name__ == '__main__':
    main()
