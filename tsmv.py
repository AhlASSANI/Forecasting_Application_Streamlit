import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from scipy import stats
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.api import VECM
from statsmodels.tsa.api import acf, graphics, pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
import matplotlib
import streamlit as st
import base64


matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

def tsmv():
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

    st.title("Multivariate time series")

###########################################################################################################

    data = st.file_uploader("Upload a Dataset", type=["csv"])


    if data is not None:
        df = pd.read_csv(data,sep=";")
        st.dataframe(df)
    
    

    st.sidebar.markdown("# Times series models")

    model = st.sidebar.selectbox("Choose the type of model you want",['Introduction','VAR','VECM','VARMAX','DFM'])
    image1="https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/error1.png"

    if model == "Introduction":
        st.info(""" - Please be sure that you have imported your data before trying to use this part of the application.
        
                     Otherwise, you will encounter this error down below 👇.
                    """)
        st.image(image1)

    if model == "VAR":
        st.sidebar.markdown("## Vector Autoregressive Model")
    
        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)   
            y_n = st.sidebar.multiselect('Endogeneous Variable', options=numeric_columns)
            y=df[y_n]
            lags = st.sidebar.slider("Chose the lag lengh of the model", min_value=1,max_value=50,value=1)
            
            if st.checkbox("Estimate the model"):
                model = VAR(y)
                results = model.fit(maxlags=lags)
                #st.table(st.write(results.summary()))
                fig=results.plot_acorr()
                st.pyplot(fig) 
                
                st.markdown("### Granger Causality test")
                caused=st.sidebar.selectbox("The caused variable in the model", options=numeric_columns)
                causing=st.sidebar.multiselect("The causing variables",options=numeric_columns)
                tg=results.test_causality(caused, causing, kind='f')
                st.table(tg.summary())
                st.info("""
                         The Null hypothesis for grangercausalitytests is that the time series in the second column, x2, does NOT Granger                            cause the time series in the first column, x1. 
                         Grange causality means that past values of x2 have a statistically significant effect on the current value of x1,                           taking past values of x1 into account as regressors. We reject the null hypothesis that x2 does not Granger cause                          x1 if the pvalues are below a desired size of the test.
                         """)
                st.sidebar.warning("""
                               Please be aware that the variable you put on both sides of
                               the granger causality tests are in the estimated model.
                                """) 
                
            if st.checkbox("Estimate the impulse response function"):
                
                model = VAR(y)
                results = model.fit(maxlags=lags)
                hIR=st.sidebar.slider("Impulse response horizon", min_value=1,max_value=100, value=1)
                irf = results.irf(hIR)
                hv=st.sidebar.slider("orthogonalized or non-orthogonalized IRF", min_value=0,max_value=1, value=0)
                figIR=irf.plot(orth=True)
                st.pyplot(figIR)
                
                if st.checkbox("Compute the IRF for only one variable of interest"):
                    
                    st.markdown("### Vizualize the variable of interest")               
                    INT=st.sidebar.selectbox("The variable of interest", options=numeric_columns)
                    figIN=irf.plot(impulse=INT)
                    st.pyplot(figIN)
                
            if st.checkbox("Estimate the forecast error variance"):
                model = VAR(y)
                results = model.fit(maxlags=lags)
                fevd = results.fevd(lags)
                hv=st.sidebar.slider("Forecast error variance horizon", min_value=1,max_value=100, value=1)
                figv=results.fevd(hv).plot()
                st.pyplot(figv)
            if st.checkbox("Perform out of sample forecast"):
                model = VAR(y)
                results = model.fit(maxlags=lags)
                h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)       
                fig1=results.plot_forecast(h,plot_stderr=False)
                st.pyplot(fig1)
            
        except Exception as e:
            st.exception(e)

    if model == "VECM":
        st.sidebar.markdown("## Vector Error Correction Model")
    
        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            y_n = st.sidebar.multiselect('Endogeneous Variable', options=numeric_columns)
            y=df[y_n]
            lags = st.sidebar.slider("Chose the lag lengh of the model", min_value=1,max_value=50,value=1)
        
        
            if st.checkbox("Cointegration and causality test"):
            
                det_ordr = st.sidebar.slider("Deterministic order", min_value=-1,max_value=1,value=-1)
                st.sidebar.text("""
                                -1 - no deterministic terms

                                 0 - constant term

                                 1 - linear trend
                                 """
                                   )
                coint=coint_johansen(y, det_ordr,lags)
            
                Trace=coint.trace_stat
                Trace_stat=coint.trace_stat_crit_vals
             
                eigV=coint.max_eig_stat
                eigVstat=coint.max_eig_stat_crit_vals
                       
                        
                st.markdown("### Trace statistic test")
            
                t1 = {'Trace statistic':Trace,'Trace statistic critical value at 90%': Trace_stat[:,0]}
                t2= {'Trace statistic':Trace,'Trace statistic critical value at 95%': Trace_stat[:,1]}
                t3 = {'Trace statistic':Trace,'Trace statistic critical value at 99%': Trace_stat[:,2]}
            
                Trace_test1=pd.DataFrame(t1,index=y_n)
                Trace_test2=pd.DataFrame(t2,index=y_n)
                Trace_test3=pd.DataFrame(t3,index=y_n)
            
                st.write(Trace_test1)
                st.write(Trace_test2)
                st.write(Trace_test3)
            
                st.markdown("### Eigen-Value statistic test")
            
                e1 = {'Trace statistic':eigV,'Trace statistic critical value at 90%': eigVstat[:,0]}
                e2= {'Trace statistic':eigV,'Trace statistic critical value at 95%': eigVstat[:,1]}
                e3 = {'Trace statistic':eigV,'Trace statistic critical value at 99%': eigVstat[:,2]}
            
                eig_test1=pd.DataFrame(e1,index=y_n)
                eig_test2=pd.DataFrame(e2,index=y_n)
                eig_test3=pd.DataFrame(e3,index=y_n)
            
                st.dataframe(eig_test1)
                st.write(eig_test2)
                st.write(eig_test3)
            
                def make_line():
                    line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                    unsafe_allow_html=True)
                    return line
                
                make_line()
            
                st.markdown("### Granger Causality test")
                model1 = VECM(y,exog=None, exog_coint=None, k_ar_diff=lags)
                results1 = model1.fit(method='ml')
            
                caused=st.sidebar.selectbox("The caused variable in the model", options=numeric_columns)
                causing=st.sidebar.multiselect("The causing variables",options=numeric_columns)
            
                tg=results1.test_granger_causality(caused, causing)
                st.table(tg.summary())
                st.sidebar.warning("""
                                   Please be aware that the variable you put on both sides 
                                   of the granger causality tests are in the estimated model.
                                    """) 
            
            if st.checkbox("Estimate the model"):
                model1 = VECM(y,exog=None, exog_coint=None, k_ar_diff=lags)
                results1 = model1.fit(method='ml')
                results1.summary()
                st.write(results1.summary())
            
            if st.checkbox("Estimate the impulse response function"):
                
                model1 = VECM(y,exog=None, exog_coint=None, k_ar_diff=lags)
                results1 = model1.fit(method='ml')
                hIR=st.sidebar.slider("Impulse response horizon", min_value=1,max_value=100, value=1)
                irf = results1.irf(periods=hIR)
                hv=st.sidebar.slider("orthogonalized or non-orthogonalized IRF", min_value=0,max_value=1, value=0)
                figIR=irf.plot(orth=True)
                st.pyplot(figIR)
                
                if st.checkbox("Compute the IRF for only one variable of interest"):
                    st.markdown("### Vizualize the variable of interest")               
                    INT=st.sidebar.selectbox("The variable of interest", options=numeric_columns)
                    figIN=irf.plot(impulse=INT)
                    st.pyplot(figIN)
                
          
            if st.checkbox("Perform out of sample forecast"):
                model1 = VECM(y,exog=None, exog_coint=None, k_ar_diff=lags)
                results1 = model1.fit(method='ml')
                nblast=st.sidebar.slider("Number of last observation to include in the graph", min_value=0,max_value=len(y), value=1)
                h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                confint=st.sidebar.slider("Choose to include whether or not the forecast confidence interval",
                                         min_value=0,max_value=1, value=0)
                fig1=results1.plot_forecast(h, plot_conf_int=confint,n_last_obs=nblast)
                st.pyplot(fig1)
                st.set_option('deprecation.showPyplotGlobalUse', False)
            
        except Exception as e:
            st.exception(e)

    
    if model == "VARMAX":
        st.sidebar.markdown("## State space estimation of VAR model")

        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            st.sidebar.markdown("## State space Vector Autoregressive Model")
            y_n = st.sidebar.multiselect('Endogeneous Variable', options=numeric_columns)
            y=df[y_n]
        #x_n = st.sidebar.multiselect('Exogeneous Variable', options=numeric_columns)
        #x=df[x_n]
            MA=st.sidebar.slider("Order of the model for the number of MA parameters to use", min_value=0,max_value=20, value=1)
            AR=st.sidebar.slider("Order of the model for the number of AR parameters to use", min_value=0,max_value=20, value=1)
    
   
            if st.checkbox("Estimate the model"):
                mod = sm.tsa.VARMAX(y, order=(AR,MA))
                res = mod.fit()
                st.write(res.summary())
                Vdiag = st.sidebar.selectbox('Select the variable you want to plot the diagnosis', options=numeric_columns)
                lagsF = st.sidebar.slider("The lag for the key feature of the results",
                                                          min_value=1,max_value=100,value=1)
                fig = plt.figure(figsize=(16, 16))
                fig = res.plot_diagnostics(variable=Vdiag,fig=fig, lags=lagsF)
                st.pyplot(fig)
            if st.checkbox("perform out of sample forecast"):
                mod = sm.tsa.VARMAX(y, order=(AR,MA))
                res = mod.fit()
                h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                yfor=res.forecast(steps=h, signal_only=False)
                yfors=res.get_forecast(steps=h, signal_only=False)
                yfors_ci=yfors.conf_int()
            
                Vfor = st.sidebar.selectbox('Select the variable you want to plot the forecast', options=numeric_columns)
            
                Yfp=df[Vfor]
                fig = plt.figure(figsize=(9, 7))
                ax = Yfp.plot(label='observed')
                ax.fill_between(yfors_ci.index,yfors_ci['lower'+' '+str(Vfor)],yfors_ci['upper'+' '+str(Vfor)], color='g', alpha=.4)
                tfy=yfor[Vfor]
                plt.plot(tfy)
                st.pyplot(fig)
                                
                fig1 = plt.figure(figsize=(9, 7))
                ax = tfy.plot(label='observed')
                ax.fill_between(yfors_ci.index,yfors_ci['lower'+' '+str(Vfor)],yfors_ci['upper'+' '+str(Vfor)], color='g', alpha=.4)
                st.pyplot(fig1)
        except Exception as e:
            st.exception(e)
            
    if model == "DFM":
        st.sidebar.markdown("## Dynamic factor model with EM algorithm")
        try:
            numeric_columns = list(df.select_dtypes(['float','int']).columns)
            non_numeric_columns = list(df.select_dtypes(['object']).columns)
            y_n = st.sidebar.multiselect('Endogeneous Variable', options=numeric_columns)
            y=df[y_n]
            if st.checkbox("Estimate the model"):
                mod_dfm = sm.tsa.DynamicFactorMQ(y)
                res_dfm = mod_dfm.fit()
                st.write(res_dfm.summary())
                fig_dfm = res_dfm.plot_coefficients_of_determination("individual")
                fig_dfm.suptitle('$R^2$ - regression on individual factors', fontsize=14, fontweight=600)
                st.pyplot(fig_dfm)
                Vdiag = st.sidebar.selectbox('Select the variable you want to plot the diagnosis', options=numeric_columns)
                lagsF = st.sidebar.slider("The lag for the key feature of the results",min_value=1,max_value=100,value=1)
                fig = plt.figure(figsize=(16, 16))
                fig = res_dfm.plot_diagnostics(variable=Vdiag,fig=fig, lags=lagsF)
                st.pyplot(fig)
 
            if st.checkbox("perform out of sample forecast"):
                mod_dfm1 = sm.tsa.DynamicFactorMQ(y)
                res = mod_dfm1.fit()
                h=st.sidebar.slider("H-step ahead forecast", min_value=1,max_value=100, value=1)
                yfor=res.forecast(steps=h, signal_only=False)
                yfors=res.get_forecast(steps=h, signal_only=False)
                yfors_ci=yfors.conf_int()
            
                Vfor = st.sidebar.selectbox('Select the variable you want to plot the forecast', options=numeric_columns)
            
                Yfp=df[Vfor]
                fig = plt.figure(figsize=(9, 7))
                ax = Yfp.plot(label='observed')
                ax.fill_between(yfors_ci.index,yfors_ci['lower'+' '+str(Vfor)],yfors_ci['upper'+' '+str(Vfor)], color='g', alpha=.4)
                tfy=yfor[Vfor]
                plt.plot(tfy)
                st.pyplot(fig)
                                
                fig1 = plt.figure(figsize=(9, 7))
                ax = tfy.plot(label='observed')
                ax.fill_between(yfors_ci.index,yfors_ci['lower'+' '+str(Vfor)],yfors_ci['upper'+' '+str(Vfor)], color='g', alpha=.4)
                st.pyplot(fig1)
        except Exception as e:
            st.exception(e)

    return 
    







