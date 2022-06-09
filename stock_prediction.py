from select import select
import streamlit as st
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

from prophet import Prophet
# import prophet.plot as plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = datetime.today().strftime("%Y-%m-%d")#date.today().strftime("%Y-%m-%d")

st.title("Stock Predction")

# later use investpy to select stocks by country
stocks = ("AAPL", "GOOG", "GME", "NFLX")
selected_stocks = st.selectbox("Select stock to predict : ", stocks)

n_years = st.slider("Years of prediction: ", 1, 4)
period = n_years*365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data ...")
data = load_data(selected_stocks)
data_load_state.text("Loading data ..._done!")

st.subheader(" Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Open'],
            name='Stock Open'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            name='Stock Close'
        )
    )

    fig.layout.update(
        title_text="Time Series Data ",
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig)

plot_raw_data()

#Forecasting
data_load_state = st.text("Load Forecast ...")

df_train = data[['Date', 'Close']]

df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)

forecast = m.predict(future)

st.subheader(" Forecast Data")
st.write(forecast.tail())

st.write("Forecast data")
fig1 = m.plot(forecast)
st.plotly_chart(fig1)

st.write("forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

data_load_state.text("Loading forecast ..._done!")

