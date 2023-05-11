import numpy as np
import numpy.matlib

import pyroomacoustics as pra
import plotly.graph_objects as go
import plotly.express as px

import geo_utils as geo
import mic_utils as arr

from pyroomacoustics.directivities import cardioid_func

def compute_weights_freq(d, R):
    invRd = np.matmul(np.linalg.pinv(R),d)
    return invRd/np.matmul(np.conj(d).T,invRd)


def mvdr(svect, Rn_fii):
    assert svect.shape[:2] == Rn_fii.shape[:2]
    n_rfft, n_mics = svect.shape
    weights = np.zeros((n_rfft, n_mics), dtype=np.complex128)  # nFreqs x nMics
    for f in range(n_rfft):
        weights[f,:] = compute_weights_freq(svect[f,:], Rn_fii[f,:,:])
    return weights


def delay_and_sum(svect):
    # BEAMFORMER
    n_rfft, n_mics = svect.shape
    Rn = np.eye(n_mics)
    weights = np.zeros((n_rfft, n_mics), dtype=np.complex128)  # nFreqs x nMics
    for f in range(n_rfft):
        weights[f,:] = compute_weights_freq(svect[f,:], Rn)
    return weights


def get_quadrature_weights(dirs):
    # DESCRIPTION: returns the quadrature weights for the elevations 
    # NOTE: These weights are needed to compensate for higher density of points closer to the poles due to uniform grid sampling of full-space
    inc = (np.pi/2)-dirs[:,1] # [nDir x 1]
    Ninc = np.size(np.unique(dirs[:,1]))
    Nazi = np.size(np.unique(dirs[:,0]))
    m = np.arange(Ninc/2)
    m = m[:,None].T # (1,M)
    w = np.sum(np.sin(inc[:,None] * (2*m+1)) / (2*m+1),axis=1) * np.sin(inc) * 2 / (Ninc*Nazi) #(nDir,1)
    w = w / np.sum(w)
    return w


def diag_load_cov(R, diag_load_mode ='cond', diag_load_val = int(1e2)):
    # DESCRIPTION: diagonal loading of covariance matrix
    # *** INPUTS ***
    # R (ndarray)  covariance matrix [nChan x nChan]
    nChan = R.shape[0]
    
    if diag_load_mode:
        R = R + diag_load_val * np.eye(nChan)
    else:
        cn0 = np.linalg.cond(R) # original condition number
        threshold = diag_load_val 
        if cn0>threshold:
            ev = np.linalg.eig(R)[0] # eigenvalues only
            R = R + np.eye(nChan) * (ev.max() - threshold * ev.min()) / (threshold-1)
    return R


def compute_iso_weights(svect, dirs):
    nFreq, nDirs, nChan = svect.shape       # [nFreq x nDir x nChan]
    w_quad = get_quadrature_weights(dirs)   # [nDir x 1]
    w_quad = np.matlib.tile(w_quad,(nFreq,nChan,nChan,1))   # [nFreq x nChan x nChan x nDir]
    w_quad = np.transpose(w_quad,axes=[0,3,1,2])            # [nFreq x nDir x nChan x nChan]

    R_iso = svect[:,:,:,None] @ np.conj(svect[:,:,None,:])      # [nFreq x nDir x nChan x nChan]
    R_iso = R_iso * w_quad # quadrature weighting
    R_iso = np.squeeze(np.sum(R_iso,axis = 1))  # [nFreq x nChan x nChan]
    for fi in range(nFreq):
        R_iso[fi,:,:] = diag_load_cov(np.squeeze(R_iso[fi,:,:]))
    return R_iso


def compute_all_steering_vect(mic_pos, ref_pos, doas, freqs, c=343.):
    az = doas[:,0]
    el = doas[:,1]
    vect_mics = -(mic_pos - ref_pos)
    vect_doa = np.stack([
        np.cos(el)*np.cos(az),
        np.cos(el)*np.sin(az),
        np.sin(el)
    ], axis=-1)
    toas_far_free = (vect_doa @ vect_mics) / c                                          # nDoas x nMics
    svect = np.exp(- 1j * 2 * np.pi * freqs[:,None,None] * toas_far_free[None,:,:])     # nFreqs x nDoas x nMics
    return svect


def plot_beampatten(beamforming='delay_and_sum', doa_deg=60, fquery=500, fquery_delta=1, 
                    array_type='linear', phi=0, n_mics=4, spacing=0.1):

    # HPARAMS
    Fs = 48000
    speed_of_sound = 343.
    n_rfft = 1025

    # TARGET POSITION
    target_azi = np.deg2rad(doa_deg)
    target_ele = np.deg2rad(0.)
    target_distance = 2

    target_pol = np.array([[target_distance, target_azi, target_ele]]).T
    target_pos = geo.sph2cart(target_pol, deg=False)

    # ARRAY GEOMETRY
    array_center = np.zeros([3,1])
    array_prop = {"spacing" : spacing, "n_mics" : n_mics, "phi" : phi, "n_rfft" : n_rfft, "fs" : Fs}
    mic_pos =  arr.get_array(array_type, array_center, array_prop)
    mic_pos_polar = geo.cart2sph(mic_pos, deg=False)
    n_mics = mic_pos.shape[1]

    ref_pos = np.zeros([3,1])
    freqs = np.linspace(0, 1, n_rfft) * (Fs / 2)


    # BEAMFORMER
    toas = np.linalg.norm(target_pos - mic_pos, axis=0, keepdims=True) / speed_of_sound     # [1 x nMics]
    svect = np.exp(- 1j * 2 * np.pi * freqs[:,None] * toas)                                 # [nFreqs x nMics]

    if beamforming == 'delay_and_sum':
        weights = delay_and_sum(svect)
        Rn = np.eye(n_mics)
        Rn = np.matlib.tile(Rn,(n_rfft,1,1))   # [nFreq x nChan x nChan]
    
    elif beamforming == 'isotropic_mvdr':
        el = np.array([0.])
        az = np.deg2rad(np.arange(0, 360))
        doas = np.stack(np.meshgrid(az, el, indexing='ij'), axis=-1).reshape(-1,2)

        all_svect = compute_all_steering_vect(mic_pos, ref_pos, doas, freqs)     # [nFreqs x nDirs x nMics]
        Rn = compute_iso_weights(all_svect, doas)                             # [nFreqs x nMics x nMics]
        weights = mvdr(svect, Rn)

    else:
        raise ValueError(f' "{beamforming}" beamformer does not exist')


    ## BEAMPATTERN FOR ALL

    # all possible direction
    n_doas = 360
    el = np.array([0.])
    az = np.deg2rad(np.linspace(0,360,n_doas))
    doas = np.stack(np.meshgrid(az, el, indexing='ij'), axis=-1).reshape(-1,2)
    all_svect = compute_all_steering_vect(mic_pos, ref_pos, doas, freqs)     # [nFreqs x nDirs x nMics]

    # compute beampattern
    B = np.zeros((n_rfft, n_doas))

    for f in range(n_rfft):
        B[f,:] = np.abs(np.sum(weights.conj()[f,None,...] * all_svect[f,...], axis=-1))**2

    B_abs = np.abs(B)**2
    B_abs = B_abs / np.max(B_abs)

    # diplay freqs ranges
    freqs = np.linspace(0, Fs/2, n_rfft)
    beampattern = dict()
    idx_f100 = np.argmin(np.abs(freqs - 100))
    idx_f8k = np.argmin(np.abs(freqs - 8000))
    idx_fquery_min = max(np.argmin(np.abs(freqs - (fquery - fquery_delta))), 0)
    idx_fquery_max = min(np.argmin(np.abs(freqs - (fquery + fquery_delta))), 8000)
    
    beampattern["0-100"] = np.mean(B_abs[:idx_f100,:], axis=0)
    beampattern["100-8k"] = np.mean(B_abs[idx_f100:idx_f8k+1,:], axis=0)
    beampattern["all"] =  np.mean(B_abs, axis=0)
    beampattern["narrow"] =  np.mean(B_abs[idx_fquery_min:idx_fquery_max+1,:], axis=0)

    noise_cov = dict()
    Rn = np.abs(Rn)**2
    noise_cov["0-100"] = np.mean(Rn[:idx_f100,:,:], axis=0)
    noise_cov["100-8k"] = np.mean(Rn[idx_f100:idx_f8k+1,:,:], axis=0)
    noise_cov["all"] =  np.mean(Rn, axis=0)
    noise_cov["narrow"] =  np.mean(Rn[idx_fquery_min:idx_fquery_max+1,:,:], axis=0)

    # PLOT
    fig = go.Figure()
    
    for key in beampattern:
        fig.add_trace(go.Scatterpolar(
            r = beampattern[key],
            theta = np.rad2deg(az),
            mode = 'lines',
            name = f'Beampattern {key}',
        ))
    fig.add_trace(go.Scatterpolar(
            r = np.array([1.1]),
            theta = np.rad2deg(np.array([target_azi])),
            mode = 'markers',
            marker = {"size" : 10*np.ones_like(mic_pos_polar[0,:])},
            name = 'Source DOA'
    ))
    fig.add_trace(go.Scatterpolar(
            r = mic_pos_polar[0,:],
            theta = np.rad2deg(mic_pos_polar[1,:]),
            marker = {"size" : 10*np.ones_like(mic_pos_polar[0,:])},
            mode = 'markers',
            name = 'Array'
    ))

    fig.add_trace(go.Scatterpolar(
            r = np.ones_like(az),
            theta = np.rad2deg(az),
            mode = 'lines',
            name = 'Unit resposne',
            line_color = 'darkviolet',
            opacity=0.1,
    ))

    fig.update_layout(
        title = 'Beam Patterns',
        showlegend = True,
        height=800,
        polar=dict(radialaxis=dict(range=[0, 1.1]))
    )
    fig_beampattern = fig


    # ## Fig Noise covariance
    fig = px.imshow(noise_cov["all"], color_continuous_scale='gray',
                    x = [str(i+1) for i in range(n_mics)],
                    y = [str(i+1) for i in range(n_mics)],
                    labels=dict(x="Mics", y="Mics", color="Scale"))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig_noise_cov = fig
    
    return fig_beampattern, fig_noise_cov


if __name__ ==  "__main__":   
    fig = plot_beampatten(60)
    fig.show()
