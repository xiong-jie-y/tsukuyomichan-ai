import json
import os
import time

import numpy as np
import simpleaudio as sa
import speech_recognition as sr
import yaml
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from tsukuyomichan_talksoft import TsukuyomichanTalksoft

from bert_agent import EmbeddingBasedReplyAgent

config = yaml.load(open(os.path.expanduser("~/.tsukuyomichanai.yaml")))

r = sr.Recognizer()
speech = sr.Microphone()
authenticator = IAMAuthenticator(config['watson']['api_key'])
speech_to_text = SpeechToTextV1(
    authenticator=authenticator,
)

speech_to_text.set_service_url(config['watson']['url'])

talksoft = TsukuyomichanTalksoft(model_version='v.1.2.0')
MAX_WAV_VALUE = 32768.0
fs = 24000

agent = EmbeddingBasedReplyAgent("conversation_dir")
if not agent.is_ready():
    agent.ingest_csv("conversation_pair.txt")

while True:
    with speech as source:
        print("say something!!…")
        audio_file = r.adjust_for_ambient_noise(source)
        audio_file = r.listen(source)

        print("start recognization")

        speech_recognition_results = speech_to_text.recognize(
            audio=audio_file.get_wav_data(), content_type='audio/wav', model="ja-JP_BroadbandModel").get_result()
        sentence = ""
        for result in speech_recognition_results["results"]:
            best_one = result["alternatives"][0]["transcript"]
            sentence += best_one

        print(sentence)

        reply = agent.reply_to(sentence)
        wav = talksoft.generate_voice(reply, 1)
        wav = wav * MAX_WAV_VALUE
        wav = wav.astype(np.int16)
        sa.play_buffer(wav, 1, 2, fs)

        time.sleep(1)
