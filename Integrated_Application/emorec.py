#Author: Eren Ali Aslangiray, Mehmet Enis İşgören

import librosa
import numpy as np
from scipy.signal import resample
from keras.models import load_model
import speech_recognition as sr
import fasttext as ft
import pywt
import joblib
from collections import Counter
from functools import reduce
from itertools import groupby


# 0 = neutral, 1 = angry, 2 = happy, 3 = sad


class Transformation:
    def __init__(self):
        self.multilabel_restored = ft.load_model("Models/multilabel_emotion.bin", label_prefix='__label__')
        self.binary_restored = ft.load_model("Models/binary_emotion.bin", label_prefix='__label__')

        self.mfccs_model = load_model('Models/mfcc_7k.h5')
        self.db_model = load_model('Models/db_7k.h5')
        self.wavelet_model = load_model('Models/wavelet_7k.h5')
        self.zero_model = load_model('Models/Zero_7K')

        self.mfccs_model_small = load_model('Models/mfcc_final_model.h5')
        self.db_model_small = load_model('Models/Db_final_model.h5')
        self.wavelet_model_small = load_model('Models/Wavelet_final_model.h5')
        self.zero_model_small = load_model('Models/Zero_Small')

        self.mfccs_model_forest = joblib.load('Models/Random_Forest_Mfcc')
        self.db_model_forest = joblib.load('Models/Random_Forest_Db')
        self.wavelet_model_forest = joblib.load('Models/Random_Forest_Wavelet')
        self.zero_model_forest = joblib.load('Models/Random_Forest_Zero')

        self.default_steps = {'mfcc': self.mfcc_calculator,'db': self.db_calculator,'wavelet': self.wavelet_calculator,
                              'zero_cross': self.zero_cross_rate}

        self.models = {'mfcc': self.mfccs_model,
                       'db': self.db_model,
                       'wavelet': self.wavelet_model,
                       'zero_cross': self.zero_model,
                       "mfccsmall":self.mfccs_model_small,
                       "dbsmall":self.db_model_small,
                       "waveletsmall":self.wavelet_model_small,
                       "zero_crosssmall":self.zero_model_small,
                       "mfcc_forest":self.mfccs_model_forest,
                       "db_forest":self.db_model_forest,
                       "wavelet_forest":self.wavelet_model_forest,
                       "zero_forest":self.zero_model_forest}

        self.labeldict = {"sad": 3, "neutral": 0, "anger": 1, "happy": 2,"fear":1, "surprise":2}

    def transformations(self, input_file1= "speech_audio_normalized.wav",input_file2 = "recognizer.wav"):
        x, sample_rate = librosa.load(input_file1)
        text = self.speechRecognizer(input_file2)
        self.text_results =  self.fast_text(text)
        self.voice_results = []

        for i in range(len(self.models)):
            print('____{}_____'.format([list(self.models.keys())[i]]))
            if i > 3:
                m = i-4
                if i >7:
                    m = i-8
                    p = self.default_steps[list(self.default_steps.keys())[m]](x,var=0)
                    print(self.models[list(self.models.keys())[i]].predict(p)[0])
                    self.voice_results.append(self.models[list(self.models.keys())[i]].predict(p)[0])
                else:
                    p = self.default_steps[list(self.default_steps.keys())[m]](x)
                    print(self.models[list(self.models.keys())[i]].predict_classes(p)[0])
                    self.voice_results.append(self.models[list(self.models.keys())[i]].predict_classes(p)[0])
            else:
                m = i
                p = self.default_steps[list(self.default_steps.keys())[m]](x)
                print(self.models[list(self.models.keys())[i]].predict_classes(p)[0])
                self.voice_results.append(self.models[list(self.models.keys())[i]].predict_classes(p)[0])
        results = self.voice_text_combiner(self.voice_results,self.text_results)
        print(results)
        return x,results

    @staticmethod
    def mfcc_calculator(x, sr=22050, var=1):
        o1 = np.asanyarray(np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T,axis=0))
        o1 = np.expand_dims(o1, axis=0)
        if var == 1:
            return np.expand_dims(o1, axis=2)
        else:
            return o1

    @staticmethod
    def zero_cross_rate(file_, scale_to = 120, var=1):
        if var == 1:
            array = librosa.feature.zero_crossing_rate(file_)[0]
            array2return = np.asarray(resample(x=array, num=scale_to))
            return np.expand_dims(np.expand_dims(array2return, axis=2), axis=0)
        else:
            array = librosa.feature.zero_crossing_rate(file_)[0]
            array2return = np.asarray(resample(x=array, num=scale_to))
            array2return = np.expand_dims(array2return, axis=0)
            return array2return


    @staticmethod
    def wavelet_calculator(X,sample_rate=128, var=1):
        a, b = pywt.dwt(X, 'db1')
        b = []
        for item in a:
            if item > 0.001:
                b.append(item)
        a = []
        k = 0
        sums = 0
        for item in b:
            if len(a) == sample_rate:
                continue
            sums = item + sums
            k = k + 1
            if k == int(len(b) / sample_rate):
                a.append(sums / k)
                k = 0
                sums = 0
        o2 = np.asanyarray(a)
        o2 = np.expand_dims(o2, axis=0)
        if var == 1:
            return np.expand_dims(o2, axis=2)
        else:
            return o2


    @staticmethod
    def db_calculator(X, sr=22050, var=1):
        S = librosa.feature.melspectrogram(y=X, sr=sr)
        dbmean = np.mean(librosa.amplitude_to_db(S, ref=np.max).T, axis=0)
        o3 = np.asanyarray(dbmean)
        o3 = np.expand_dims(o3, axis=0)
        if var == 1:
            return np.expand_dims(o3, axis=2)
        else:
            return o3


    @staticmethod
    def speechRecognizer(file_name='recognizer.wav'):
        recognizer = sr.Recognizer()
        with sr.WavFile(file_name) as source:
            audio = recognizer.record(source)
        print(recognizer.recognize_google(audio))
        return [recognizer.recognize_google(audio)]

    @staticmethod
    def calculator(listin, neg_rate, p):
        whitelist = ["happy", "neutral"]
        blacklist = ["anger", "sad"]
        pos_rate = 1.0 - neg_rate
        if pos_rate > neg_rate:
            pos_rate = pos_rate * 10 * p / 2
            neg_rate = neg_rate * 10 / p / 2
        else:
            pos_rate = pos_rate * 10 / p / 2
            neg_rate = neg_rate * 10 * p / 2
        ailist = []
        for i in range(len(listin)):
            if listin[i][0] in whitelist:
                ailist.append(listin[i][1] * pos_rate)
            else:
                ailist.append(listin[i][1] * neg_rate)
        sumup = sum(ailist)
        for i in range(len(ailist)):
            ailist[i] = (ailist[i] * 100 / sumup)
        finallist = []
        for i in range(len(listin)):
            finallist.append((listin[i][0], ailist[i]))
        return finallist


    def fast_text(self, text,penalty=1.0):
        labels = self.multilabel_restored.predict_proba(text, k=4)
        labels_b = self.binary_restored.predict_proba(text, k=2)
        labels = labels[0]
        labels_b = labels_b[0]
        if penalty == 0:
            return labels
        if labels_b[0][1] == labels_b[1][1]:
            m1_buffed = labels
        if labels_b[0][0] == "negative":
            negrate = labels_b[0][1]
        else:
            negrate = labels_b[1][1]
        m1_buffed = self.calculator(labels, negrate, penalty)
        print (m1_buffed)
        return m1_buffed

    @staticmethod
    def reduceByKey(func, iterable):
        get_first = lambda p: p[0]
        get_second = lambda p: p[1]
        return map(
            lambda l: (l[0], reduce(func, map(get_second, l[1]))),
            groupby(sorted(iterable, key=get_first), get_first))


    def voice_text_combiner(self, voice_results, text_results):
        arr = list(Counter(voice_results).items())
        voice_prob = []
        for item in arr:
            voice_prob.append((item[0], item[1] / 12))
        textlabeled = []
        for item in text_results:
            textlabeled.append((self.labeldict[item[0]], int(item[1]) * 0.01))
        merged_list = list(self.reduceByKey(lambda x, y: x + y, list(Counter(voice_prob + textlabeled))))
        q = 0
        for item in merged_list:
            if q < item[1]:
                q = item[1]
        for item in merged_list:
            if item[1] == q:
                return ((item[0],item[1]/2))


#a = Transformation()
#a.transformations("speech_audio_normalized.wav","recognizer.wav")
