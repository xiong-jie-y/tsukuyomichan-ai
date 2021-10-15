import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer
import simpleaudio as sa
import speech_recognition as sr
import yaml
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from tsukuyomichan_talksoft import TsukuyomichanTalksoft

from bert_agent import EmbeddingBasedReplyAgent
from yamnet import HumanVoiceDetector

import librosa

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text


class TsukuyomichanAgent:
    MAX_WAV_VALUE = 32768.0
    fs = 24000

    def __init__(self):
        self.r = sr.Recognizer()
        self.speech = sr.Microphone(sample_rate=16000)
        self.talksoft = TsukuyomichanTalksoft(model_version='v.1.2.0')
        self.voice_detector = HumanVoiceDetector()
        self.sentence_transformer = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

        d = ModelDownloader()
        self.speech2text = Speech2Text(
                **d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave"),
                device="cuda"
        )

    def speak(self, reply, seed=1):
        wav = self.talksoft.generate_voice(reply, seed)
        wav = wav * self.MAX_WAV_VALUE
        wav = wav.astype(np.int16)
        play_obj = sa.play_buffer(wav, 1, 2, self.fs)
        play_obj.wait_done()

    def listen_voice_event(self):
        return self.voice_detector.wait_for_human_voice()

    def recognize_talk(self, timeout=None):
        with self.speech as source:
            print("start listening")
            audio_file = self.r.adjust_for_ambient_noise(source)
            try:
                audio_file = self.r.listen(source, timeout=timeout)
            except sr.WaitTimeoutError:
                return None

            print("start recognization")

            sentence = self.speech2text(librosa.util.buf_to_float(audio_file.get_wav_data(), n_bytes=2, dtype=np.int16))[0][0]
            print(sentence)
            return sentence

    def wait_for_one_of_in_similar_meaning(self, sentences, timeout=None):
        user_talk = self.recognize_talk(timeout)
        if not user_talk:
            return None

        user_embedding = self.sentence_transformer.encode(user_talk)
        distances = [np.linalg.norm(user_embedding - self.sentence_transformer.encode(sentence)) for sentence in sentences]
        min_index = np.argmin(distances)

        MAX_ACCEPTABLE_DISTANCE = 0.6
        if distances[min_index] < MAX_ACCEPTABLE_DISTANCE:
            return sentences[min_index]

        return None

    def speech_to_text(self, wave):
        nbests = self.speech2text(wave)
        text, *_ = nbests[0]
        return text

import inspect
import importlib

agent = TsukuyomichanAgent()
conversations = []

module = importlib.import_module("conversations.basic_conversations")
for _, obj in inspect.getmembers(module):
    if inspect.isclass(obj) and inspect.getmodule(obj) == module:
        conversations.append(obj(agent))

while True:
    event = agent.listen_voice_event()

    for conversation in conversations:
        if conversation.react_to(event):
            conversation.start(event, agent)
            break
