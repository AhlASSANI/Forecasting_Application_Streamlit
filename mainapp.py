
import streamlit as st


from page_fit import page_fit
from page_introduction import page_introduction
from TS_univariate import TS_univariate
from tsmv import tsmv
import numpy as np


st.set_page_config(page_title='Forecasting Applications',page_icon="🌍")


def main():


    pages = {
        "Introduction": page_introduction,
        "Fit distributions": page_fit,
        "Univariate Time series": TS_univariate,
        "Multivariate Time series":tsmv,
    }

    st.sidebar.title("Main options")

    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    pages[page]()

    

if __name__ == "__main__":
    main()
