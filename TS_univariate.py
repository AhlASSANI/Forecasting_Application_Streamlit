import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import matplotlib
import plotly.express as px

import streamlit as st
import base64



def TS_univariate():
    

    matplotlib.rcParams['axes.labelsize'] = 14
    matplotlib.rcParams['xtick.labelsize'] = 12
    matplotlib.rcParams['ytick.labelsize'] = 12
    matplotlib.rcParams['text.color'] = 'k'


    @st.experimental_memo
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


    @st.experimental_memo
    def get_img_as_base64(file):
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()


    img = get_img_as_base64("image.jpg")

    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] > .main {{
    background-image: url("https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/appim.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    [data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{img}");
    background-position: 180; 
    background-repeat: no-repeat;
    background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"] {{
    right: 2rem;
    }}
    </style>
    """



    st.markdown(page_bg_img, unsafe_allow_html=True)

    st.title("Univariate time series")


    data = st.file_uploader("Upload a Dataset", type=["csv"])


    if data is not None:
        df = pd.read_csv(data,sep=";")
        st.dataframe(df)
       
    st.sidebar.markdown("# Times series models")
    image1="https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/error1.png"
    
    model = st.sidebar.selectbox("Select the univariate time serie model",["Introduction","AR", "ARMA",
                                                                           "Filters","MS Dynamic Regression","MS-AR","UC"])
    if model == "Introduction":
        st.info(""" - Please be sure that you have imported your data before trying to use this part of the application.
        
                     Otherwise, you will encounter this error down below 👇.
                    """)
        st.image(image1)
    if model == "AR":
        st.sidebar.markdown("# Autoregressions")
        specAR=st.sidebar.selectbox("Autoregressive options",["Standard AR(p)","AR(p) with seasonal dummies"])
    
        if specAR == "Standard AR(p)":
         
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                lags = st.sidebar.slider("The autogressive order of the model", min_value=1,max_value=50,value=1)

                y_n = st.sidebar.selectbox('Endogeneous Variable', options=numeric_columns)
                y=df[y_n]
                if st.checkbox("Estimate the model"):
                    
                    mod = AutoReg(y, lags, old_names=False)
                    res = mod.fit()
                    st.write(res.summary())
                    lagsF = st.sidebar.slider("Choose the lag for the key feature of the results",min_value=1,max_value=100,value=1)
                    fig = plt.figure(figsize=(16, 14))
                    fig = res.plot_diagnostics(fig=fig, lags=lagsF)
                    st.pyplot(fig)
                    
                if st.checkbox("Out of sample forecast"):
                    T=len(y)
                    st.sidebar.markdown("# Forecast horizon")
                    h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                    mod = AutoReg(y, lags, old_names=False)
                    res = mod.fit()
                    fig = res.plot_predict(start=T, end=T+h,figsize=(16, 12))
                    st.pyplot(fig)
            except Exception as e:
                st.exception(e)
        
                                
        if specAR == "AR(p) with seasonal dummies":
                                
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                lags = st.sidebar.slider("The autogressive order of the model", min_value=1,max_value=50,value=1)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Endogeneous Variable', options=numeric_columns)
                y=df[y_n]

                
                if st.checkbox("Estimate the model"):
                    
                    sel = ar_select_order(y, lags, glob=True, old_names=False)
                    sel.ar_lags
                    res = sel.model.fit()
                    st.write(res.summary())
                    st.write(res.summary())
                    lagsF = st.sidebar.slider("Lag order for the key feature of the results",min_value=1,max_value=100,value=1)
                    fig = plt.figure(figsize=(16, 14))
                    fig = res.plot_diagnostics(fig=fig, lags=lagsF)
                    st.pyplot(fig)
                if st.checkbox("Out of sample forecast"):
                    T=len(y)
                    h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                    sel = ar_select_order(y, lags, glob=True, old_names=False)
                    sel.ar_lags
                    res = sel.model.fit()
                    fig = res.plot_predict(start=T, end=T+h,figsize=(16, 14))
                    st.pyplot(fig)
                
            except Exception as e:
                 st.exception(e)
                                
    if model == "ARMA":
        specARMA=st.sidebar.selectbox("Please choose the autoregressive specifications",["ARMA model","ARIMA model","SARIMAX model"])
        if specARMA == "ARMA model":
            st.sidebar.markdown("# Autoregressive Integrated Moving Average")
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Endogeneous Variable', options=numeric_columns)
                y=df[y_n]
                lags = st.sidebar.slider("The order of the AR component", min_value=1,max_value=50,value=1)
                Mags = st.sidebar.slider("The order of the MA component", min_value=0,max_value=50,value=0)
                
                if st.checkbox("Estimate the model"):
                    mod = sm.tsa.arima.ARIMA(y, order=(lags, 0, Mags))
                    res = mod.fit()
                    st.write(res.summary())
                    lagsF = st.sidebar.slider("The lag for the key feature of the results", min_value=1,max_value=100,value=1)
                    fig = plt.figure(figsize=(16, 14))
                    fig = res.plot_diagnostics(fig=fig, lags=lagsF)
                    st.pyplot(fig)
                if st.checkbox("Out of sample forecast"):
                    T=len(y)
                    st.sidebar.markdown("# Forecast horizon")
                    h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                    mod = sm.tsa.arima.ARIMA(y, order=(lags, 0, Mags))
                    res = mod.fit()
                    pred_Arma = res.forecast(steps=h)
                    pred_Arma2 = res.get_forecast(steps=h)
                    predarma_ci=pred_Arma2.conf_int()
                    fig = plt.figure(figsize=(16, 14))
                    ax = y.plot(label='observed')
                    ax.fill_between(predarma_ci.index,predarma_ci.iloc[:, 0],predarma_ci.iloc[:, 1], color='k', alpha=.45)
                    plt.plot(pred_Arma)
                    
                    
                    st.pyplot(fig)
                    fig1 = plt.figure(figsize=(16, 14))
                    ax = pred_Arma.plot(label='observed')
                    ax.fill_between(predarma_ci.index,predarma_ci.iloc[:, 0],predarma_ci.iloc[:, 1], color='k', alpha=.45)
                    st.pyplot(fig1)
            except Exception as e:
                st.exception(e)
    
    
    
        if specARMA == "ARIMA model":
            st.sidebar.markdown("# Autoregressive Integrated Moving Average")
            
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Endogeneous Variable', options=numeric_columns)
                y=df[y_n]
                st.sidebar.markdown("# Design of the model")
                lags = st.sidebar.slider("The order of the AR component", min_value=1,max_value=50,value=1)
                Mags = st.sidebar.slider("The order of the MA component", min_value=0,max_value=50,value=0)
                Dags = st.sidebar.slider("The value of the integration component", min_value=0,max_value=50,value=0)
            
                if st.checkbox("Estimate the model"):
                    mod = sm.tsa.arima.ARIMA(y, order=(lags, Dags, Mags))
                    res = mod.fit()
                    st.write(res.summary())
                    lagsF = st.sidebar.slider("Choose the lag for the key feature of the results",min_value=1,max_value=100,value=1)
                    fig = plt.figure(figsize=(16, 14))
                    fig = res.plot_diagnostics(fig=fig, lags=lagsF)
                    st.pyplot(fig)
                    
                if st.checkbox("Out of sample forecast"):
                    T=len(y)
                    st.sidebar.markdown("# Forecast horizon")
                    h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                    mod = sm.tsa.arima.ARIMA(y, order=(lags, Dags, Mags))
                    res = mod.fit()
                    pred_Arma = res.forecast(steps=h)
                    pred_Arma2 = res.get_forecast(steps=h)
                    predarma_ci=pred_Arma2.conf_int()
                    fig = plt.figure(figsize=(9, 7))
                    ax = y.plot(label='observed')
                    ax.fill_between(predarma_ci.index,predarma_ci.iloc[:, 0],predarma_ci.iloc[:, 1], color='k', alpha=.45)
                    plt.plot(pred_Arma)
                    st.pyplot(fig)
                    fig1 = plt.figure(figsize=(9, 7))
                    ax = pred_Arma.plot(label='observed')
                    ax.fill_between(predarma_ci.index,predarma_ci.iloc[:, 0],predarma_ci.iloc[:, 1], color='k', alpha=.45)
                    st.pyplot(fig1)
            except Exception as e:
                 st.exception(e)
    
    
        if specARMA == "SARIMAX model":
            
            st.sidebar.markdown("# Seasonal Autoregressive Integrated Moving Average")
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Y axis', options=numeric_columns)
                y=df[y_n]
                p=st.sidebar.slider("p", min_value=0,max_value=20, value=0)
                d=st.sidebar.slider("d", min_value=0,max_value=20, value=0)
                q=st.sidebar.slider("q", min_value=0,max_value=20, value=0)
                st.sidebar.markdown("# Seasonal order")
                P=st.sidebar.slider("P", min_value=0,max_value=50, value=0)
                D=st.sidebar.slider("D", min_value=0,max_value=50, value=0)
                Q=st.sidebar.slider("Q", min_value=0,max_value=50, value=0)
                M=st.sidebar.slider("M", min_value=0,max_value=50, value=0)
    
                if st.checkbox("Estimate the model"):
                    mod = sm.tsa.statespace.SARIMAX(y,order=(p, d, q),
                                                seasonal_order= (P,D,Q,M),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                    results = mod.fit()
                    st.write(results.summary())
                    st.pyplot(results.plot_diagnostics(figsize=(16, 14)))
            
                if st.checkbox("Out of sample forecast"):
                    st.sidebar.markdown("# Forecast horizon")
                    h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                    mod = sm.tsa.statespace.SARIMAX(y,order=(p, d, q),
                                                seasonal_order=(P,D,Q,M),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                    results = mod.fit()
                    pred_Arma = results.forecast(steps=h)
                    pred_Arma2 = results.get_forecast(steps=h)
                    predarma_ci=pred_Arma2.conf_int()
                    fig = plt.figure(figsize=(16, 14))
                    ax = y.plot(label='observed')
                    ax.fill_between(predarma_ci.index,predarma_ci.iloc[:, 0],predarma_ci.iloc[:, 1], color='k', alpha=.45)
                    plt.plot(pred_Arma)        
                    st.pyplot(fig)

            except Exception as e:
                st.exception(e)
    
    if model == "Filters":
        specFilters=st.sidebar.selectbox("Select the filters you want to apply to your data",
                                         ["Christiano-Fitzgerald Filter","Baxter-King Filter","Hodrick-Prescott Filter"])       
   
    
        if specFilters == "Christiano-Fitzgerald Filter":
            st.sidebar.markdown("## Christiano-Fitzgerald band-pass Filter") 

            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Y axis', options=numeric_columns)
                y=df[y_n]
                H=st.sidebar.slider("High filters frequency", min_value=1,max_value=100, value=32)
                L=st.sidebar.slider("Low filters frequency", min_value=1,max_value=100, value=6)
                cf_cycles, cf_trend = sm.tsa.filters.cffilter(y, low=L, high=H, drift=True)
            
                if st.checkbox("Plot the output"):                
                    fig, axes = plt.subplots(2,figsize=(16, 14))
                    ax = axes[0]
                    ax.plot(cf_cycles)
                    ax.set_xlim(y.index[4], y.index[-1])
                    ax.set(title="Cyclical component")

                    ax = axes[1]
                    ax.plot(cf_trend)
                    ax.set_xlim(y.index[4], y.index[-1])
                    ax.set(title="Trend component")
                    st.pyplot(fig)
            except Exception as e:
                st.exception(e)
            
        
        if specFilters == "Baxter-King Filter":
            st.sidebar.markdown("## Baxter-King band-pass Filter")
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Y axis', options=numeric_columns)
                y=df[y_n]
                H=st.sidebar.slider("Maximum period for oscillations", min_value=1,max_value=100, value=32)
                L=st.sidebar.slider("Minimum period for oscillations", min_value=1,max_value=100, value=6)
                K=st.sidebar.slider("Lead-lag length of the filter", min_value=1,max_value=100, value=12)

                cf_cycles= sm.tsa.filters.bkfilter(y,L,H, K)
            
                if st.checkbox("Plot the output"):                
                    fig, axes = plt.subplots(1,figsize=(16, 14))
                    ax = axes
                    ax.plot(cf_cycles)
                    ax.set_xlim(y.index[4], y.index[-1])
                    ax.set(title="Cyclical component")
                    st.pyplot(fig)
            except Exception as e:
                st.exception(e)
    
        if specFilters == "Hodrick-Prescott Filter":       
            st.sidebar.markdown("## Hodrick-Prescott Filter")
            try:
                numeric_columns = list(df.select_dtypes(['float','int']).columns)
                non_numeric_columns = list(df.select_dtypes(['object']).columns)
                y_n = st.sidebar.selectbox('Y axis', options=numeric_columns)
                y=df[y_n]
    
                L=st.sidebar.slider("Lambda parameter", min_value=100,max_value=20000, value=1600)
                cf_cycles, cf_trend = sm.tsa.filters.hpfilter(y, L)
            
                if st.checkbox("Plot the output"):                
                    fig, axes = plt.subplots(2,figsize=(16, 14))
                    ax = axes[0]
                    ax.plot(cf_cycles)
                    ax.set_xlim(y.index[4], y.index[-1])
                    ax.set(title="Cyclical component")
                    ax = axes[1]
                    ax.plot(cf_trend)
                    ax.set_xlim(y.index[4], y.index[-1])
                    ax.set(title="Trend component")
                    st.pyplot(fig)
            except Exception as e:
                st.exception(e)
        
    
    if model == "MS Dynamic Regression":
        st.sidebar.markdown("## Markov switching dynamic regression models")
        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            x_v = st.sidebar.multiselect('X axis',numeric_columns)
            y_v = st.sidebar.selectbox('Y axis',numeric_columns)
            X=df[x_v]
            Y=df[y_v]
            Trend=st.sidebar.selectbox("Please choose whether or not to include a trend",["n","c","t","ct"])
            ordr=st.sidebar.slider("The order of the model describes the dependence of the likelihood on previous regimes",
                                    min_value=0,max_value=50, value=0)
       
            if st.checkbox("Estimate the model"):
                modMSR = sm.tsa.MarkovRegression(Y, k_regimes=2, trend=Trend,
                                                 order=ordr)
                resMSR = modMSR.fit(iter=1000)
                st.write(resMSR.summary())
                fig, axes = plt.subplots(2, figsize=(16, 14))
                ax = axes[0]
                ax.plot(resMSR.filtered_marginal_probabilities[0])
                ax.set_xlim(Y.index[4], Y.index[-1])
                ax.set(title="Filtered probability")
                ax = axes[1]
                ax.plot(resMSR.smoothed_marginal_probabilities[0])
                ax.set_xlim(Y.index[4], Y.index[-1])
                ax.set(title="Smoothed probability")
                st.pyplot(fig)
                st.text('Expected regimes duration 👇')
                st.table(resMSR.expected_durations)
        except Exception as e:
            st.exception(e)
    
    if model == "MS-AR":
        st.sidebar.markdown("## Markov switching autoregression models")
    
        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            y_v = st.sidebar.selectbox('Y axis',numeric_columns)
            Y=df[y_v]
            Trend=st.sidebar.selectbox("Please choose whether or not to include a trend",["n","c","t","ct"])
            ordr=st.sidebar.slider("The order of the autoregressive lag polynomial",
                                    min_value=1,max_value=50, value=1)
       
            if st.checkbox("Estimate the model"):
            
                ModMSAR =sm.tsa.MarkovAutoregression(Y, k_regimes=2, order=ordr,switching_ar=False)
                resMSAR = ModMSAR.fit()
                st.write(resMSAR.summary())
                fig, axes = plt.subplots(2,figsize=(16, 7))
                ax = axes[0]
                ax.plot(resMSAR.filtered_marginal_probabilities[0])
                ax.set_xlim(Y.index[4], Y.index[-1])
                ax.set(title="Filtered probability")
                ax = axes[1]
                ax.plot(resMSAR.smoothed_marginal_probabilities[0])
                ax.set_xlim(Y.index[4], Y.index[-1])
                ax.set(title="Smoothed probability")
                st.pyplot(fig)
                st.text('Expected regimes duration 👇')
                st.table(resMSAR.expected_durations)
        except Exception as e:
            st.exception(e)
        

        
    if model == "UC":
        st.sidebar.markdown("## Univariate unobserved components time series model")
        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            y_v = st.sidebar.selectbox('Y axis',numeric_columns)
            Y=df[y_v]
            ordr=st.sidebar.slider("The order of the autoregressive lag polynomial",min_value=1,max_value=20, value=1)
            st.sidebar.markdown("### Optional key features of the model")
            ireg=st.sidebar.slider("Whether or not to include an irregular component.",min_value=0,max_value=1, value=0)
            lstoc=st.sidebar.slider("Whether or not any level component is stochastic.",min_value=0,max_value=1, value=0)
            tstoc=st.sidebar.slider("Whether or not any trend component is stochastic.",min_value=0,max_value=1, value=0)
            s_stoc=st.sidebar.slider("Whether or not any seasonal component is stochastic.",min_value=0,max_value=1, value=0)
            c_stoc=st.sidebar.slider("Whether or not any cycle component is stochastic.",min_value=0,max_value=1, value=0)
            c_damped=st.sidebar.slider("Whether or not the cycle component is damped.",min_value=0,max_value=1, value=0)
            difus=st.sidebar.slider("Whether or not to use exact diffuse initialization for non-stationary states.",
                                   min_value=0,max_value=1,value=0)
       
            if st.checkbox("Estimate the model"):
                mod1 = sm.tsa.UnobservedComponents(Y,irregular=ireg,autoregressive=ordr,
                                                       stochastic_level=lstoc,stochastic_trend=tstoc,stochastic_cycle=c_stoc,
                                                       use_exact_diffuse=difus,damped_cycle=c_damped,stochastic_seasonal=s_stoc)
                res1 = mod1.fit()
                lagsF = st.sidebar.slider("Choose the lag for the key feature of the results",
                                                          min_value=1,max_value=100,value=1)
                
                st.write(res1.summary())
                st.pyplot(res1.plot_diagnostics(figsize=(16, 14),lags=lagsF))
            
            if st.checkbox("Out of sample forecast"):
                st.sidebar.markdown("# Forecast horizon")
                h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1) 
                mod1 = sm.tsa.UnobservedComponents(Y,irregular=ireg,autoregressive=ordr,
                                                   stochastic_level=lstoc,stochastic_trend=tstoc,stochastic_cycle=c_stoc,
                                                   use_exact_diffuse=difus,damped_cycle=c_damped,stochastic_seasonal=s_stoc)
                res1 = mod1.fit()
            
                pred_UC = res1.forecast(steps=h)
                pred_UC2 = res1.get_forecast(steps=h)
                predUC_ci=pred_UC2.conf_int()
                
                fig = plt.figure(figsize=(16, 14))
                ax = Y.plot(label='observed')
                ax.fill_between(predUC_ci.index,predUC_ci['lower'+' '+str(y_v)],predUC_ci['upper'+' '+str(y_v)], color='k', alpha=.4)
                plt.plot(pred_UC)        
                st.pyplot(fig)
                
                fig1 = plt.figure(figsize=(16, 14))
                ax = pred_UC.plot(label='observed')
                ax.fill_between(predUC_ci.index,predUC_ci['lower'+' '+str(y_v)],predUC_ci['upper'+' '+str(y_v)], color='k', alpha=.4)
                st.pyplot(fig1)
                

        except Exception as e:
            st.exception(e)
            
            
        
    return
