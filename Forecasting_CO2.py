import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load data
df = pd.read_excel('emisi_co2_indonesia_new.xlsx')

# Ubah format data tahun
df['year'] = pd.to_datetime(df['year'], format='%Y')
df.set_index(['year'], inplace=True)

def show_page(page):
    if page == 'Modeling and Analysing Data':
        st.title('Forecasting CO2 Emissions in Indonesia : Modeling and Analysing Data using Time Series')
        # Sidebar untuk pemodelan


        # Subjudul
        st.write("""
        ## 1) Background
        """)
        # Menambahkan teks penjelasan
        st.markdown("""
        <div style="text-align: justify">
        Indonesia faces challenges in managing carbon dioxide (CO2) emissions, especially from the industrial and transportation sectors. High CO2 emissions can have significant impacts on global climate change as well as local air quality, with the potential to threaten human health and environmental sustainability.<br>
        
        In this context, the forecasting model of CO2 emissions in the future is important. The forecasting model can provide valuable information for stakeholders, researchers, and the public in planning mitigation strategies, optimizing resources, and estimating the impacts of policies to reduce CO2 emissions.<br>
        
        By understanding the historical trends of CO2 emissions, we can develop the forecasting model to predict the level of CO2 emissions in Indonesia in the future. It will be a contribution to the mitigation of climate change and the protection of air quality in the future.
        <br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: justify">
        Emissions data are sourced from: 
        <a href="https://data.world/makeovermonday/2019w22/workspace/file?filename=CO2+emissions+per+capita+per+country.csv">https://data.world</a>
        and <a href="https://data.worldbank.org/indicator/EN.ATM.CO2E.PC?end=2020&locations=ID&start=1990">https://data.worldbank.org</a>.
        The data will be used is CO2 emissions in Indonesia from 1960 to 2020.
        <br>
        <br>
        </div>
        """, unsafe_allow_html=True)

        st.write("""
        ## 2) Data Exploration & Data Visualisation
        """)
        # Visualize data
        st.line_chart(df)
        st.markdown("""
        <div style="text-align: justify">
        CO2 emissions in Indonesia tended to increase from 1960 to 2020 (although in 2020 it decreased from the previous year).
        <br>
        <br>
        </div>
        """, unsafe_allow_html=True)


        # Decomposition Plot
        decompose_add = seasonal_decompose(df['CO2'])
        fig = decompose_add.plot()
        st.pyplot(fig)

        st.markdown("""
        <div style="text-align: justify">
        <b>Trend:</b> Overall increase in CO2 emissions from year to year.<br>
        <b>Seasonal:</b> The seasonal graph is a straight line. It shows that there is no significant seasonal pattern in the data. So, CO2 emissions do not show consistent periodic or cyclical changes over time. <br>
        <b>Residual:</b> The residual is around the zero line. It indicates that the long-term trends and seasonal patterns that have been identified are good enough to explain most of the variation in the data.
        <br>
        <br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: justify">
        
        <br>
        </div>
        """, unsafe_allow_html=True)

        # ADF Test
        def adf_test(timeseries):
            st.write('Result of Augmented Dickey-Fuller Test')
            st.write('--------------------------------')
            adftest = adfuller(timeseries)
            adf_output = pd.Series(adftest[0:4], index=['Test Statistic', 'p-value', 'Lags Used', 'Number of Observation Used'])
            for key, Value in adftest[4].items():
                adf_output['Critical Value (%s)' %key] = Value
            st.write(adf_output)
        
        st.markdown("""
        <div style="text-align: justify">
        <b>Is the Data stationary?
        <br>
        </div>
        """, unsafe_allow_html=True)

        

        # Difference Plot
        diff_df = df.diff()
        diff_df.dropna(inplace=True)
        st.line_chart(diff_df)
        adf_test(diff_df.values)

        st.markdown("""
        <div style="text-align: justify">
        Since the p-value (0.1074) is greater than the significance level (e.g. 0.05), we fail to reject the null hypothesis.
        The null hypothesis in the ADF test is that the data is non-stationary.
        So, the conclusion from this result is that there is not enough evidence to state that the time series data is stationary.

        <br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: justify">
        Time Series methods can be used for non-stationary data: ARIMA (Autoregressive Integrated Moving Average), Exponential Smoothing Models (including Single Exponential Smoothing, Double Exponential Smoothing), etc.

        <br>
        </div>
        """, unsafe_allow_html=True)

        
        train_df = df.loc[:'2002-01-01']
        test_df = df.loc['2003-01-01':]
        
        st.write("""
        ## 3) Modeling Data
        """)
        # Single Exponential Smoothing
        st.write("""
        ### Single Exponential Smoothing
        """)
        single_exp = SimpleExpSmoothing(train_df).fit()
        single_exp_train_pred = single_exp.fittedvalues
        single_exp_test_pred = single_exp.forecast(18)

        fig, ax = plt.subplots()
        train_df['CO2'].plot(style='--', color='gray', legend=True, label='train_df')
        test_df['CO2'].plot(style='--', color='r', legend=True, label='test_df')
        single_exp_test_pred.plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)

        # Double Exponential Smoothing
        st.write("""
        ### Double Exponential Smoothing
        """)
        double_exp = ExponentialSmoothing(train_df, trend=None, initialization_method='heuristic', seasonal='add', seasonal_periods=15, damped_trend=False).fit()
        double_exp_train_pred = double_exp.fittedvalues
        double_exp_test_pred = double_exp.forecast(18)
        
        fig, ax = plt.subplots()
        train_df['CO2'].plot(style='--', color='gray', legend=True, label='train_df')
        test_df['CO2'].plot(style='--', color='r', legend=True, label='test_df')
        double_exp_test_pred.plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)

        # ARIMA Model
        st.write("""
        ### ARIMA
        """)
        ar = ARIMA(train_df, order=(19,1,19)).fit()
        ar_train_pred = ar.fittedvalues
        ar_test_pred = ar.forecast(18)

        fig, ax = plt.subplots()
        train_df['CO2'].plot(style='--', color='gray', legend=True, label='train_df')
        test_df['CO2'].plot(style='--', color='r', legend=True, label='test_df')
        ar_test_pred.plot(color='b', legend=True, label='Prediction')
        st.pyplot(fig)

        # Model Comparison
        st.write("""
        ### Model Comparison
        """)
        comparison_df = pd.DataFrame(data=[
            ['Single Exp Smoothing', mean_squared_error(test_df, single_exp_test_pred)**0.5, mean_absolute_percentage_error(test_df, single_exp_test_pred)],
            ['Double Exp Smoothing', mean_squared_error(test_df, double_exp_test_pred)**0.5, mean_absolute_percentage_error(test_df, double_exp_test_pred)],
            ['ARIMA', mean_squared_error(test_df, ar_test_pred)**0.5, mean_absolute_percentage_error(test_df, ar_test_pred)]
            ],
            columns=['Model', 'RMSE', 'MAPE'])
        comparison_df.set_index('Model', inplace=True)
        st.write("Model Comparison", comparison_df.sort_values(by='RMSE'))

        st.markdown("""
        <div style="text-align: justify">
        Based on the graph above and the RMSE & MAPE values, it can be concluded that the best model for forecasting CO2 emissions in Indonesia is <b>ARIMA</b>. Then, we use this model in forecasting CO2 emissions in Indonesia for other years.

        <br>
        </div>
        """, unsafe_allow_html=True)


    elif page == 'Forecasting Data':

        # Fungsi untuk memprediksi dengan ARIMA
        def forecast_CO2(df, years):
            ar_future = ARIMA(df, order=(19,1,19)).fit()
            ar_future_pred = ar_future.forecast(years)
            return ar_future_pred

        # Membuat aplikasi Streamlit
        st.write("""
        ## Forecasting CO2 emissions in the next X years in Indonesia
        """)

        # Widget input untuk memilih jumlah tahun ke depan
        years = st.number_input('Input the number of years:', min_value=1, step=1, value=25)

        # Melakukan prediksi    
        ar_future_pred = forecast_CO2(df, years)

        # Plotting hasil prediksi
        fig, ax = plt.subplots()
        df['CO2'].plot(style='--', color='gray', legend=True, label='Known')
        ar_future_pred.plot(color='b', legend=True, label='Prediction')
        plt.xlabel('Year')
        plt.ylabel('CO2 Emissions')
        st.pyplot(fig)
        
        # Membuat tabel dari hasil prediksi
        prediction_table = pd.DataFrame({'Tahun': ar_future_pred.index, 'Predicted CO2 Emissions': ar_future_pred.values.flatten()})
        st.write("### Forecasting Results of CO2 Emissions in Indonesia for the Next {} Years:".format(years))
        st.dataframe(prediction_table, height=500)

        st.write("""
        ### Insight & Recommendation
        """)

        st.markdown("""
        <div style="text-align: justify">
        Based on the forecasting model, CO2 emissions in Indonesia tend to increase from year to year. This indicates that there is continued growth in emission levels. So, it's crucial to address this issue to mitigate the potential negative impacts on climate change. 
        <br></br>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align: justify">
        Here are some strategies to tackle the increase in CO2 emissions in Indonesia:<br>
        <b>(1) Development of Renewable Energy <br>
        (2) Enhance Energy Efficiency <br>
        (3) Public Transportation and Sustainable Mobility <br>
        (4) Strengthen Environmental Regulations <br>
        (5) Encourage Afforestation and Reforestation</b>

        <br>
        </div>
        """, unsafe_allow_html=True)


# Daftar halaman yang dapat dipilih di sidebar
pages = ['Modeling and Analysing Data', 'Forecasting Data']

# Sidebar untuk navigasi halaman
st.sidebar.title('Forecasting CO2 Emissions in Indonesia')
selected_page = st.sidebar.radio('Pilih Halaman', pages)

# Menampilkan konten di halaman utama berdasarkan pilihan di sidebar
show_page(selected_page)