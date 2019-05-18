import keras
import librosa
import numpy as np
import pywt

def modelloader():
    #Output order: mfcc, db, wavelet.
    mfccs_model = keras.models.load_model('./Models/mfcc_final_model.h5')
    db_model = keras.models.load_model('./Models/Db_final_model.h5')
    wavelet_model = keras.models.load_model('./Models/Wavelet_final_model.h5')
    return mfccs_model, db_model, wavelet_model


def soundtransformations (inputwav):
    #Output order: mfcc, meandb, wavelet.
    X, sample_rate = librosa.load(inputwav)
    if sample_rate != 22050:
        raise Exception("Sample rate must be 22050. Check your sample rate!")
    #getting mfccs
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    #getting mean db
    S = librosa.feature.melspectrogram(y=X, sr=sample_rate)
    dbmean = np.mean(librosa.amplitude_to_db(S, ref=np.max).T,axis=0)
    #wavelet
    a,b = pywt.dwt(X, 'db1')
    b = []
    for item in a:
        if item > 0.001:
            b.append(item)
    a = []
    k = 0
    sums = 0
    for item in b:
        if len(a) == 128:
            continue
        sums = item + sums
        k = k+1
        if k == int(len(b)/128):
            a.append(sums/k)
            k=0
            sums=0
    if len(a) != 128 or len(mfccs) != 40 or len(dbmean) != 128:
        raise Exception("Error: Check input data or soundtransformations() function.")
    o1,o2,o3 =  np.asanyarray(mfccs) , np.asanyarray(dbmean), np.asanyarray(a)
    o1,o2,o3 =  np.expand_dims(o1, axis=0), np.expand_dims(o2, axis=0), np.expand_dims(o3, axis=0)
    return np.expand_dims(o1, axis=2), np.expand_dims(o2, axis=2), np.expand_dims(o3, axis=2)

modelloader()