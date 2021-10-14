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
from yamnet import HumanVoiceDetector

import librosa

r = sr.Recognizer()
speech = sr.Microphone(sample_rate=16000)

talksoft = TsukuyomichanTalksoft(model_version='v.1.2.0')
MAX_WAV_VALUE = 32768.0
fs = 24000

agent = EmbeddingBasedReplyAgent("conversation_dir")
if not agent.is_ready():
    agent.ingest_csv("conversation_pair.txt")

voice_detector = HumanVoiceDetector()

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

# 学習済みをダウンロードし、音声認識モデルを作成
d = ModelDownloader()
speech2text = Speech2Text(
        **d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave"),
        device="cuda"  # CPU で認識を行う場合は省略
    )

trigger_words = ["ねぇ", "あ", "つくよみちゃん", "助けて", "困った"]

while True:
    wave = voice_detector.wait_for_human_voice()

    # 認識結果の取得と表示
    nbests = speech2text(wave)
    text, *_ = nbests[0]

    print(f"you said {text}")
    if text not in trigger_words:
        continue

    time.sleep(0.3)

    wav = talksoft.generate_voice("何?", 1)
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype(np.int16)
    sa.play_buffer(wav, 1, 2, fs)

    with speech as source:
        print("start listening")
        audio_file = r.adjust_for_ambient_noise(source)
        audio_file = r.listen(source)

        print("start recognization")

        sentence = speech2text(librosa.util.buf_to_float(audio_file.get_wav_data(), n_bytes=2, dtype=np.int16))[0][0]
        print(sentence)

        reply = agent.reply_to(sentence)

        wav = talksoft.generate_voice(reply, 1)
        wav = wav * MAX_WAV_VALUE
        wav = wav.astype(np.int16)
        sa.play_buffer(wav, 1, 2, fs)

        time.sleep(1)
