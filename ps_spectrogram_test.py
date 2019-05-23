import matplotlib as mlp
mlp.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import madmom
import tensorflow as tf
from scipy.io import savemat
import ps_preprocessing
from pathlib import Path


# val = 2
# m = tf.placeholder(tf.int32)
#
# m_feed = [[0, 1, val,   val, val],
#           [0,   0,   val, val,   val],
#           [0  , 1,   val,   val,   val]]
# tmp_indices = tf.where(tf.equal(m, val))
# result = tf.segment_min(tmp_indices[:, 1], tmp_indices[:, 0])
# with tf.Session() as sess:
#     print(sess.run(result, feed_dict={m: m_feed}))

audio_config = {'num_channels': 1,
                'sample_rate': 44100,
                'frame_size': 4096,
                'fft_size': 4096,
                'fps': 100,
                'num_bands': 48,
                'fmin': 10.0,
                'fmax': 5000.0,
                'fref': 440.0,
                'norm_filters': True,
                'unique_filters': True,
                'circular_shift': False,
                'norm': True}

base_dir = Path('../../MAPS/ENSTDkAm/MUS/').resolve()
file = 'MAPS_MUS-chpn_op25_e4_ENSTDkAm'
spec = ps_preprocessing.wav_to_spec(str(base_dir), file, audio_config)
#ohpcp = ps_preprocessing.wav_to_hpcp(str(base_dir), file)

#spec_proc = madmom.audio.spectrogram.LogarithmicFilteredSpectrogramProcessor()

#spec = spec_proc('../../MAPS/ENSTDkAm/MUS/MAPS_MUS-chpn_op25_e3_ENSTDkAm.wav', **audio_config)
#spec = spec_proc('../../MAPS/ENSTDkCl/MUS/MAPS_MUS-liz_rhap09_ENSTDkCl.wav', **audio_config)
#spec = spec_proc('../../00_BN1-129-Eb_comp_mic.wav', **audio_config)
#superflux_proc = madmom.audio.spectrogram.SpectrogramDifferenceProcessor(diff_max_bins=3)
#superflux_freq = superflux_proc(spec.T)
#superflux_freq = superflux_freq.T

#superflux_time = superflux_proc(spec)


#comb = spec+superflux_time +superflux_freq
#print(np.max(comb))
#comb = comb/np.max(comb)
#comb = np.clip(comb, a_min=0.001, a_max=1.0)
#print(comb.shape)
plt.imshow(spec[0:1000, :].T, aspect='auto', origin='lower')

#savemat("MAPS_MUS-chpn_op25_e4_ENSTDkAm_spec", {"spec": spec[:555, :], "ohpcp": ohpcp[:555, :]})
savemat("MAPS_MUS-chpn_op25_e4_ENSTDkAm_spec", {"spec": spec})

plt.show()