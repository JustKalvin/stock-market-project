import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go 
import base64

def get_base64(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode()

# Ganti path ke gambar kamu
image_base64 = get_base64("./stonk.jpg")

# Masukin ke CSS
page_bg_color = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""

st.markdown(page_bg_color, unsafe_allow_html=True)


sidebar_base64 = get_base64('./stonk_sidebar.png')

sidebar_bg_image = f"""
<style>
[data-testid="stSidebar"] {{
    background-image: url("data:image/jpeg;base64,{sidebar_base64}");
    background-size: cover;
    background-position: center;
}}
</style>
"""

st.markdown(sidebar_bg_image, unsafe_allow_html=True)



st.title('Stock Market Prediction')

date_input_style = """
<style>
[data-testid="stSidebar"] [data-testid="stDateInput"] input {
    background-color: #1E1E1E; /* Warna background */
    color: white; /* Warna teks */
    border-radius: 8px; /* Radius border */
    padding: 8px; /* Padding */
    border: 1px solid #555; /* Warna border */
}
</style>
"""

st.markdown(date_input_style, unsafe_allow_html=True)

text_input_style = """
<style>
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
    background-color: #1E1E1E; /* Warna background */
    color: white; /* Warna teks */
    border-radius: 8px; /* Radius border */
    padding: 8px; /* Padding */
    border: 1px solid #555; /* Warna border */
}
</style>
"""

st.markdown(text_input_style, unsafe_allow_html=True)

selectbox_style = """
<style>
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    background-color: #1E1E1E; /* Warna background */
    color: white; /* Warna teks */
    border-radius: 8px; /* Radius border */
    padding: 8px; /* Padding */
    border: 1px solid #555; /* Warna border */
}

/* Styling teks di dalam selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div {
    color: white !important;
}
</style>
"""

st.markdown(selectbox_style, unsafe_allow_html=True)


# Input from sidebar
ticker = st.sidebar.text_input('Code Saham', 'BBCA.JK')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
period_options = {
    '1 Day' : '1d',
    '5 Days' : '5d',
    '1 Month' : '1mo',
    '3 Months' : '3mo',
    '6 Months' : '6mo',
    '1 Year' : '1y',
    '5 Years' : '5y'
}

selected_period = st.sidebar.selectbox("Pilih Rentang Waktu", list(period_options.keys()))
prediction_days_num = int(st.sidebar.text_input('Prediksi N Hari Kedepan', 1))
years = (end_date - start_date).days // 365
if years < 5:
    st.warning("Untuk hasil yang lebih baik, rentang waktu harus minimal 5 tahun")
else :   
    # Download stock data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)

    # Reset index (just in case)
    data = data.reset_index()
    data.columns = data.columns.droplevel(1)

    # Display data summary
    st.subheader(f'Stock Data From {start_date} To {end_date}')
    st.write(data) 

    # Closing Price vs Time Chart with 100MA
    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = data['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], ma100, label='100-Day Moving Average', color='red')
    plt.title(f'{ticker} Closing Price and 100-Day MA')
    plt.legend(loc='best')
    st.pyplot(fig)

    # Closing Price vs Time Chart with 200MA
    st.subheader('Closing Price vs Time Chart with 200MA')
    ma200 = data['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='blue')
    plt.plot(data['Date'], ma100, label='100-Day Moving Average', color='red')
    plt.plot(data['Date'], ma200, label='200-Day Moving Average', color='green')
    plt.title(f'{ticker} Closing Price, 100-Day and 200-Day MAs')
    plt.legend(loc='best')
    st.pyplot(fig)

    # Candlestick
    data_candle = yf.download(ticker, period=period_options[selected_period], interval="1d")
    data_candle = data_candle.reset_index()  # Reset index agar 'Date' menjadi kolom biasa
    data_candle.columns = data_candle.columns.droplevel(1)

    # Candlestick Chart
    st.subheader(f'Candlestick Chart ({selected_period})')

    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=data_candle['Date'],
                open=data_candle['Open'],
                high=data_candle['High'],
                low=data_candle['Low'],
                close=data_candle['Close'],
                name='Candlestick'
            )
        ]
    )

    fig_candle.update_layout(
        title=f'{ticker} Candlestick Chart ({selected_period})',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    # Tampilkan chart di Streamlit
    st.plotly_chart(fig_candle)

    train_data = pd.DataFrame(data['Close'][0:int(len(data) * 0.7)])
    test_data = pd.DataFrame(data['Close'][int(len(data) * 0.7) : len(data)])

    scaler = MinMaxScaler(feature_range = (0, 1))
    train_data_array = scaler.fit_transform(train_data)

    # Splitting data with sliding windows
    x_train, y_train = [], []
    for i in range(100, train_data_array.shape[0]) :
        x_train.append(train_data_array[i - 100 : i])
        y_train.append(train_data_array[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Load model
    model = load_model('./Final_Model.h5')

    past_100_days = train_data.tail(100)
    final_df = pd.concat([past_100_days, test_data], ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test, y_test = [], []

    for i in range(100, input_data.shape[0]) :
        x_test.append(input_data[i - 100 : i])
        y_test.append(input_data[i, 0])
    x_test, y_test = np.array(x_test), np.array(y_test)

    y_pred = model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()


    # Final Graph
    st.subheader('Prediction VS Original')
    fig2 = plt.figure(figsize = (12, 6))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_pred, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    st.pyplot(fig2)

    # Prediction
    st.subheader('Prediction Result : ')
    data2 = yf.download(ticker, period='100d')
    
    data2 = data2.reset_index()
    data2 = data2.drop(columns = ['Date'], axis = 1)
    data2.columns = data2.columns.droplevel(1)
    data2.columns.name = None
    # st.write(data2)
    train_data2 = pd.DataFrame(data2['Close'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data2 = scaler.fit_transform(train_data2)
    train_data2 = train_data2.reshape(1, 100, 1)  # (batch_size, timesteps, features)
    n = prediction_days_num
    temp_pred = []
    from keras.models import load_model
    model = load_model('./SMPmodel.h5') 

    for i in range(n) :
        y_pred = model.predict(train_data2[:, i : train_data2.shape[1], :])
        train_data2 = np.concatenate((train_data2, y_pred.reshape(1, 1, 1)), axis=1)
        temp_pred.append(scaler.inverse_transform(y_pred).reshape(-1))
    
    temp_pred2 = temp_pred.copy()
    
    temp_pred2 = pd.DataFrame(temp_pred2, columns=['Predicted Price'])
    temp_pred2.index = range(0, n)  # Memberi index 1, 2, ..., n
    temp_pred2.index.name = 'Hari'  # Nama index
    st.write(temp_pred2)

    temp_pred = pd.DataFrame(temp_pred, columns = ['Closed'])

    temp_data1 = yf.download(ticker, period = '7d')
    temp_data1 = temp_data1.reset_index()
    temp_data1.columns = temp_data1.columns.droplevel(1)
    temp_data1 = temp_data1['Close']
    temp_data1 = temp_data1.to_numpy().reshape(-1, 1)  # Pastikan jadi (N,1)
    temp_pred = temp_pred.to_numpy().reshape(-1, 1)  # Pastikan jadi (N,1)

    fussion_data = np.vstack((temp_data1, temp_pred))


    # Plotting
    st.subheader('Stock Price with Predictions')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(fussion_data, label='Stock Price & Prediction', color='blue')
    ax.axvline(x=len(temp_data1), color='red', linestyle='dashed', label='Prediction Start')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.legend()

    st.pyplot(fig)

    # DP to make a decision when to buy and when to sell.
    def max_profit_dp(prices: list, k: int):
        from functools import lru_cache
        
        n = len(prices)
        dp = {}

        def f(i, stats, k):
            if i >= n or k < 0:
                return [0, ""]
            if (i, stats, k) in dp:
                return dp[(i, stats, k)]
            
            if stats == "buy":
                a, path1 = f(i + 1, "sell", k - 1)
                a -= prices[i]
                b, path2 = f(i + 1, "buy", k)
                if a > b:
                    dp[(i, stats, k)] = [a, f"buy at {i} -> {path1}"]
                else:
                    dp[(i, stats, k)] = [b, path2]
            else:  # sell
                a, path1 = f(i + 1, "buy", k)
                a += prices[i]
                b, path2 = f(i + 1, "sell", k)
                if a > b:
                    dp[(i, stats, k)] = [a, f"sell at {i} -> {path1}"]
                else:
                    dp[(i, stats, k)] = [b, path2]

            return dp[(i, stats, k)]

        result = f(0, "buy", k)
        return result[0], result[1]

    # Ambil harga yang diprediksi
    predicted_prices = temp_pred2['Predicted Price'].tolist()

    # Misalnya kamu batasi maksimal 2 transaksi beli-jual
    max_transaksi = 100

    profit, strategy_path = max_profit_dp(predicted_prices, max_transaksi)

    # Tampilkan di Streamlit
    st.subheader("Maximal Profit Strategy")
    st.markdown(
        f"""
        <div style='background-color: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px; color: white'>
            <p><strong>Total Max Profit:</strong> {profit:.2f}</p>
            <p><strong>Strategi:</strong> {strategy_path}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


    import requests
    import json

    payload = {
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "predicted_prices": temp_pred2['Predicted Price'].tolist()
    }

    webhook_url = "https://lioness-eternal-elf.ngrok-free.app/webhook/stockanalysis"
    st.subheader("Stock Market Analysis")
    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            st.success("Data berhasil dikirim!")
            # st.json(response.json())
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Ambil base64 string dari response
            base64_str = response.json()['base64StringImage'][0]  # Asumsinya satu gambar, ambil indeks 0

            # Decode base64 menjadi bytes
            image_bytes = base64.b64decode(base64_str)

            # Buka dengan PIL dan tampilkan di Streamlit
            image = Image.open(BytesIO(image_bytes))
            st.image(image, caption="Analisis Chart", use_column_width=True)
            # Ambil konten analisis
            analysis_text = response.json()['content'][0]

            # Styling dengan latar belakang hitam transparan 50%
            analysis_text = response.json()['content'][0]

            st.markdown(
                f"""
                <div style='background-color: rgba(0, 0, 0, 0.5); color: white; padding: 20px; border-radius: 10px; font-family: Arial, sans-serif; font-size: 16px;'>
                    {analysis_text}
                </div>
                """,
                unsafe_allow_html=True
            )

            
        else:
            st.warning(f"Gagal kirim data. Status: {response.status_code}")
    except Exception as e:
        st.error(f"Terjadi error: {e}")



  # st.write(temp_data1)
  # st.write(fussion_data)