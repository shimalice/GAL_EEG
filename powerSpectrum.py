from scipy import fftpack
import numpy as np
### Signal -> Power spectrum ###

def spectrum (vector):
    ''' raw EEG dataベクトルのpower spectrum '''
    A = np.fft.fft(vector)# /len(vector)
    ps = np.abs(A)**2
    # 正のpower spectrum
    ps = ps[:len(ps)//2]
    return ps
