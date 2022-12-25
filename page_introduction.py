
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import matplotlib
import plotly.express as px

import streamlit as st
import base64


def page_introduction():
    
    
    
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
    background-image: url("https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/frontintro.jpg");
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

    st.markdown("<h2 style='text-align: center;'> Welcome To </h2>", 
                unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'> My forecasting application</h1>", 
                unsafe_allow_html=True)
     

    st.info("""
            There are three main features: \n
            - Fit distributions  
            - Univariate Time series
            - Multivariate Time series
            $←$ To start playing with the app, select an option on the 
            left sidebar.
            """)
 

    image1 = "https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/data.png"
    image2 = "https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/UV.png"
    image3 = "https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/MV.png"
    image4 = "https://raw.githubusercontent.com/AhlASSANI/Forecasting_Application_Streamlit/main/fit.png"



    
    def make_line():
        """ Line divider between images. """
            
        line = st.markdown('<hr style="border:1px solid gray"> </hr>',
                unsafe_allow_html=True)

        return line    


    # Images and brief explanations.
    feature1, feature2 = st.columns([0.5,0.4])
    with feature1:
        st.image(image1, use_column_width=True)
    with feature2:
        st.warning('Importation of the data')
        st.info("""
                - Open your Excel or CSV sheet and replace the {,} by {.}.
                - Then save your data into a CSV (UTF-8). 
                
                """)
    
    make_line()
    
    feature3, feature4 = st.columns([0.6,0.4])
    with feature3:        
        st.image(image2, use_column_width=True)
    with feature4:
        st.warning('Univariate time serie model')
        st.info("""
                The univariate time series model contains six types of models:
                 
                - Autoregressive model
                  - Standard autoregressive model
                  - An autoregressive model with seasonal dummies
                  
                - The autoregressive moving average model
                  - Standard ARMA
                  - The autoregressive integrated moving average model
                  - The seasonal autoregressive integrated moving average model (State Space estimation)
                  
                - Time series filter
                  - Christiano-Fitzgerald bandpass filter
                  - Baxter king bandpass filter
                  - HP filters 
                  
                - Markov switching dynamic regression
                
                - Markov switching autoregressive model
                """)
    make_line()
    
    feature5, feature6 = st.columns([0.5,0.5])
    with feature5:
        st.image(image3, use_column_width=True)
    with feature6:
        st.warning('Multivariate time series model')
        st.info("""
                The multivariate time series model contains four types of models:
                - The vector autoregressive model
                - The vector error correction model
                - The state-space version of the vector autoregressive model
                - The dynamic factor model (EM-Algorithm)
                """)
    
    make_line()
    
    feature7, feature8 = st.columns([0.6,0.5])
    with feature7:
        st.warning('Fit distributions')
        st.info("""
               Fit distributions contain more than 100 distribution laws from scipy:
               
               - This part of the application has been develop by [**Robert Dzudzar**](https://github.com/rdzudzar/DistributionAnalyser)
               
               - Within this page you can upload your own data and fit them with a 
                 distribution law or download a sample data which contain FX pair 
                 related to the Australian dollar.
                """)
    with feature8:
        st.image(image4, use_column_width=True)
    
    
    st.info('There are 100 continuous distribution functions  \
                from **SciPy v1.6.1** available to play with.')
        
    st.markdown("""
                
                - Abriviations:
                
                    - PDF - Probability Density Function
                
                    - CDF - Cumulative Density Function
                
                    - SF - Survival Function
                
                    - P(X<=x) - Probability of obtaining a value smaller than 
                                x for selected x.
                
                    - Empirical quantiles for a data array: 
                        Q1, Q2, Q3 respectively 0.25, 0.50, 0.75
                              
                    - $\sigma$ (Standard Deviation). On plot shades: 
                        mean$\pm x\sigma$
                        
                    - SSE - Sum of squared estimate of errors
                
                """)
    
    st.sidebar.header("About")
    st.sidebar.warning("""
                    This forecasting application was created and maintained by Ahaloudine ASSANI. If you have any requests regarding this                       application,feel free to send me a message on [**LinkedIn**](https://www.linkedin.com/in/ahaloudine-assani-7877501bb/).
                        """)

    
    return
