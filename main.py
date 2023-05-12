import streamlit as st

import numpy as np
import pyroomacoustics as pra

from beamformer import plot_beampatten



def main():
    st.write("""
    # A Beamformer Visualization App

    This app visualize some types of *data-independent* beampattern

    You can change the type of mic array and beamforming using the sidebar (top left button)

    This app is based on [`Pyroomacoustics`](https://github.com/LCAV/pyroomacoustics).
    Visualization is done with Plotly.
    """)


    target_doa = st.slider('**Target** direction of arrival [degree]', 0, 360, 33)

    st.sidebar.header("Parameters")

    st.sidebar.write("### Mic array")
    array_type = st.sidebar.selectbox(
        "Which array?",
        ('linear', 'circular', 'easycom'))
    Fs = st.sidebar.number_input('Sampling Frequency [Hz]', 1, 96000, 16000)
    
    array_radius = st.sidebar.slider('Intermic spacing',  1, 1000, 50) / 1000
    array_orient = st.sidebar.slider('Array orientation', 0, 360,  0)
    n_mics = st.sidebar.slider('N of mics', 0, 10, 4)

    st.sidebar.write("### Beamformer")
    bf_type = st.sidebar.selectbox(
        "Which beamformer?",
        ('delay_and_sum', 'max_di'))

    
    st.sidebar.write('### Narrowband beampattern config')
    fquery = st.sidebar.slider('Query frequency [Hz]', 0, Fs, 100)
    fdelta = st.sidebar.slider('Delta frequency [Hz]', 0, Fs, 1)

    beampatten_views = st.multiselect(
        'Which frequencies?',
            ['Speech [100Hz - 8kHz]', 'All', 'User-defined'],
            ['Speech [100Hz - 8kHz]'])

    fmin = max(fquery-fdelta,0)
    fmax = int(min(fquery+fdelta,Fs/2))
    st.write(f"**User-defined** narrow-band beampatter for [ {fmin} : {fmax} ] Hz")

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
                            fquery=fquery, fquery_delta=fdelta,
                            beampatten_views=beampatten_views,
                            Fs=Fs) 

    # st.pyplot(fig)
    st.plotly_chart(fig_bf, theme="streamlit", use_container_width=True, height=800)

    st.plotly_chart(fig_Rn, theme="streamlit", use_container_width=True, height=800)

    st.write("Created by [Chutlhu](https://github.com/Chutlhu)")

if __name__ == "__main__":
    main()