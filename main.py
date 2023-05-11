import streamlit as st

import numpy as np
import pyroomacoustics as pra

from beamformer import plot_beampatten



def main():
    st.write("""
    # A Beamformer Visualization App

    This app visualize some types of data-independent *beampattern*
    """)

    st.sidebar.header("User input parameters")

    array_type = st.sidebar.selectbox(
        "Which array?",
        ('linear', 'circular', 'easycom'))
    
    
    target_doa = st.sidebar.slider('Target DOAs', 0, 360, 33)
    array_radius = st.sidebar.slider('Intermic spacing',  0., 0.5, 0.05)
    array_orient = st.sidebar.slider('Array orientation', 0, 360,  0)
    n_mics = st.sidebar.slider('N of mics', 0, 10, 4)

    bf_type = st.sidebar.selectbox(
        "Which beamformer?",
        ('delay_and_sum', 'isotropic_mvdr'))
    
    fquery = st.sidebar.slider('Query frequency', 0, 8000, 100)
    fdelta = st.sidebar.slider('Delta frequency', 0, 8000, 1)

    # st.write(f"doas:      {target_doa} degree")
    # st.write(f"d_mics:    {array_radius:1.3f} meter")
    # st.write(f"Phi_mics:  {array_orient:1.0f} degree")
    # st.write(f"n_mics     {n_mics} mics")

    fig_bf, fig_Rn = plot_beampatten(bf_type, 
                          doa_deg=target_doa, 
                          array_type=array_type, 
                          phi=array_orient, 
                          n_mics=n_mics, 
                          spacing=array_radius,
                          fquery=fquery, fquery_delta=fdelta) 

    # st.pyplot(fig)
    st.plotly_chart(fig_bf, theme="streamlit", use_container_width=True, height=800)
    st.plotly_chart(fig_Rn, theme="streamlit", use_container_width=True, height=800)

if __name__ == "__main__":
    main()