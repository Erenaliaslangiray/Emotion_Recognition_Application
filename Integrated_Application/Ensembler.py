import ffmpeg_normalize
import speech_recognition as sr
from Listener import Recorder
import fasttext as ft
from Emotion_Model_Loader import modelloader, soundtransformations
from collections import Counter


class Ensembler:
    def __init__(self):
        self.mfccs_model, self.db_model, self.wavelet_model = modelloader()
        self.multilabel_restored = ft.load_model("./Models/multilabel_emotion.bin", label_prefix='__label__')
        self.binary_restored = ft.load_model("binary_emotion.bin", label_prefix='__label__')

    def analysis(self, file_name='speech_audio.wav'):
        # TODO this function needs to be called with a loop
        self.listen(file_name=file_name)
        normalized_file_name = self.normalizer(file_name)
        text = self.speechRecognizer(file_name=file_name)
        self.fast_text(text)  # Emotion Prob from text
        self.predicter(normalized_file_name)  # Emotion Prob from voice

    def listen(self, file_name='speech_audio.wav'):
        while True:
            keyboard_input = input("Press Enter For Voice Record")
            if keyboard_input == "":
                print("Speak Now:")
                Recorder.record(file_name=file_name)
    @staticmethod
    def soft_voting():
        # TODO implement soft_voting
        pass

    @staticmethod
    def emotion_recognizer():
        # TODO implement this function

        pass

    def predicter(self,inputfile):
        o1, o2, o3 = soundtransformations(inputfile)
        p1, p2, p3 = self.mfccs_model.predict_classes(o1)[0], self.db_model.predict_classes(o2)[0], self.wavelet_model.predict_classes(o3)[0]
        arr = [p1, p2, p3]
        arr = list(Counter(arr).items())
        k = 0
        q = 0
        for item in arr:
            if item[1] > k:
                k = item[1]
                q = item[0]
            else:
                continue
        if k == 1:
            q = p1
        return q

    def fast_text(self,text):
        """

        :param text:
        :return: labels as multilabel emotions
                labels_b return negative or positive probs
        """
        # TODO labels and labels_b may be combined with ensemble combiner to get one output

        labels = self.multilabel_restored.predict_proba(text, k=4)
        labels_b = self.binary_restored.predict_proba(text, k=2)
        return labels, labels_b


    @staticmethod
    def normalizer(file_):
        normalizer = ffmpeg_normalize.FFmpegNormalize()
        normalizer.add_media_file(input_file=file_,
                                  output_file='speech_audio_normalized.wav')
        normalizer.run_normalization()
        return 'speech_audio_normalized.wav'

    @staticmethod
    def speechRecognizer(file_name='speech_audio.wav'):
        recognizer = sr.Recognizer()
        with sr.WavFile(file_name) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio)
