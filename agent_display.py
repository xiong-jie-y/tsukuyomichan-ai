import enum
import emoji
from espnet_model_zoo.downloader import ModelDownloader
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
import numpy as np
from tha2.util import extract_PIL_image_from_filelike, resize_PIL_image, extract_pytorch_image_from_PIL_image, convert_output_image_from_torch_to_numpy
import tha2.poser.modes.mode_20
import PIL
import torch
import numpy
import cv2
import numpy as np
import time

# CTC segmentation
from espnet2.bin.asr_align import CTCSegmentation

import spacy
import ginza

import time

import simpleaudio as sa

from utils.file import get_model_file_from_gdrive
# play_obj = sa.play_buffer(wav, 1, 2, fs)

class BlinkController:
    def __init__(self, clock):
        self.clock = clock
        self.state_start_time = clock.time()
        self.state = "closing"

    def blink_rate(self):
        state_periods = {
            "closing": 0.1,
            "closed": 0.05,
            "opening": 0.1, 
            "wait": 2
        }
        next_states = {
            "closing": "closed",
            "closed": "opening",
            "opening": "wait",
            "wait": "closing"
        }

        duration_from_last = self.clock.time() - self.state_start_time

        # print(self.state)

        if duration_from_last > state_periods[self.state]:
            self.state_start_time = self.clock.time()
            self.state = next_states[self.state]

        if self.state == "closing":
            return duration_from_last / state_periods["closing"]
        elif self.state == "closed":
            return 1.0
        elif self.state == "opening":
            return 1.0 - duration_from_last/state_periods["opening"]
        elif self.state == "wait": 
            return 0.0

class BodyController:
    def __init__(self, clock):
        self.clock = clock
        self.state_start_time = clock.time()
        self.state = "closing"

    def control(self, pose, pose_parameters):
        state_periods = {
            "closing": 0.5,
            "closed": 0.5,
            "opening": 0.5, 
            "wait": 3
        }
        next_states = {
            "closing": "closed",
            "closed": "opening",
            "opening": "wait",
            "wait": "closing"
        }

        duration_from_last = self.clock.time() - self.state_start_time

        # print(self.state)

        if duration_from_last > state_periods[self.state]:
            self.state_start_time = self.clock.time()
            duration_from_last = 0
            self.state = next_states[self.state]

        AMPLITUDE = 0.25

        rate = None
        if self.state == "closing":
            rate = AMPLITUDE * duration_from_last / state_periods["closing"]
        elif self.state == "closed":
            rate = AMPLITUDE
        elif self.state == "opening":
            rate =  AMPLITUDE - AMPLITUDE * duration_from_last/state_periods["opening"]
        elif self.state == "wait": 
            rate = 0.0

        assert rate != None

        if rate > AMPLITUDE or rate < -AMPLITUDE:
            import IPython; IPython.embed()

        pose[0, pose_parameters.get_parameter_index("neck_z")] = rate
        # if rate > 0.0:
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_left")] = 1.0
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_right")] = 1.0
        #     pose[0, pose_parameters.get_parameter_index("eye_wink_left")] = 0.0
        #     pose[0, pose_parameters.get_parameter_index("eye_wink_right")] = 0.0
        # else:
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_left")] = 0
        #     pose[0, pose_parameters.get_parameter_index("eye_relaxed_right")] = 0

mouth_shapes = ["mouth_aaa", "mouth_iii", "mouth_uuu", "mouth_eee", "mouth_ooo", "mouth_delta"]

mouth_map = [
    ["アカガサザタダナハバマヤラワャ", "mouth_aaa"],
    ["イキギシジチジニヒビミリ", "mouth_iii"],
    ["ウクグスズツズヌフブムユルュ", "mouth_uuu"],
    ["エケゲセゼテデネヘベメレ", "mouth_eee"],
    ["オコゴソゾトドノホボモヨロヲョ", "mouth_ooo"],
    ["ン", "mouth_nnn"]
]

class MouthShapeController:
    def __init__(self, clock, time_mouth_map):
        self.clock = clock
        self.start_time = clock.time()
        self.time_mouth_map = time_mouth_map
        self.discrete_parameter = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def control(self, pose, pose_parameters):
        if self.time_mouth_map is None:
            for mouth_shape in mouth_shapes:
                pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0
            return False

        duration_from_sentence_start = (self.clock.time() - self.start_time)

        current_time_mouth = None
        for time_mouth in self.time_mouth_map:
            if time_mouth[0] < duration_from_sentence_start and duration_from_sentence_start <= time_mouth[1]:
                current_time_mouth = time_mouth
                break

        if self.time_mouth_map[-1][1] < duration_from_sentence_start:
            return True

        if current_time_mouth is None:
            for mouth_shape in mouth_shapes:
                pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0
            return False

        name = current_time_mouth[2]

        last_utterance = 0

        # print((time_mouth[1] - time_mouth[0]))

        # duration_from_start = duration_from_sentence_start - current_time_mouth[0]
        # pil_image.save(f"{name}.png")
        progress_rate_in_utterance = (duration_from_sentence_start - current_time_mouth[0]) / ((current_time_mouth[1] - current_time_mouth[0]) * 0.65)
        progress_rate_in_utterance = progress_rate_in_utterance if progress_rate_in_utterance < 1.0 else 1.0
        
        # indice = np.searchsorted([progress_rate_in_utterance], self.discrete_parameter, side="left")[0]

        # degree_rate = self.discrete_parameter[indice]
        if name is not None: #  and (self.clock.time() - last_utterance) > 0.05:
            # print(progress_rate_in_utterance)
            print(f"Controlling with {current_time_mouth}: {progress_rate_in_utterance}")
    
            if name != "mouth_nnn":
                pose[0, pose_parameters.get_parameter_index(name)] = progress_rate_in_utterance
            for mouth_shape in mouth_shapes:
                if mouth_shape != name:
                    pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0

        last_utterance = self.clock.time()
        return False

import queue
import simpleaudio as sa
import re

import onnxruntime
            
# from pyanime4k import ac
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# class UpscaleMethod(enum.Enum):
#     RealESRGAN = "realesrgan"
#     RealESRGANOnnx = "realesrgan_onnx"

class RealESRGANUpscaler:
    def __init__(self):
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=get_model_file_from_gdrive("RealESRGAN_x4plus_anime_6B.pth", "https://drive.google.com/uc?id=1cExySdxIOh0mw7XK_P_LZiijDKMx9m-p"),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False)

    def upscale(self, pil_image):
        output, _ = self.upsampler.enhance(np.array(pil_image), outscale=4)
        pil_image = PIL.Image.fromarray(output, mode='RGB')

        return pil_image

class FastSRGANUpscaler:
    def __init__(self):
        self.super_resolution_session = onnxruntime.InferenceSession("generator2.onnx", providers = ['CUDAExecutionProvider'])

    def upscale(self, pil_image):
        s = time.time()
        output_image_2 = self.super_resolution_session.run([], {"input_1": [np.array(pil_image)/255]})[0][0]
        output_image_2 = (((output_image_2 + 1) / 2.) * 255).astype(np.uint8)
        output_image_2 = cv2.resize(output_image_2, (output_image_2.shape[0] //2, output_image_2.shape[1]//2))
        pil_image = PIL.Image.fromarray(output_image_2, mode='RGB')
        # pil_image = pil_image.resize((pil_image.width // 2, pil_image.height //2 ))
        print(time.time() - s)
        return pil_image

# class RealESRGANONNXUpscaler:
#     def __init__(self):
#         self.super_resolution_session = onnxruntime.InferenceSession("RealESRGAN_x4plus_anime_6B.onnx", providers = ['CUDAExecutionProvider'])

#     def upscale(self, pil_image):
#         # import IPython; IPython.embed()
#         output_image_2 = self.super_resolution_session.run([], {"image.1": [np.transpose(np.array(pil_image), (2, 0, 1)) / 255]})[0][0]
#         pil_image2 = PIL.Image.fromarray(numpy.uint8(numpy.rint(np.transpose(output_image_2[[2, 1, 0], :, :], (1, 2, 0)) * 255)), mode='RGB')
#         return pil_image2

class TsukuyomichanVisualizationGenerator:
    def __init__(self, clock, background_color=(0,0,0), upscale=True, 
        upscale_method=RealESRGANUpscaler):
        self.device = torch.device("cuda:0")

        self.poser = tha2.poser.modes.mode_20.create_poser(self.device)
        self.pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
        self.pose_size = self.poser.get_num_parameters()
        self.pose = torch.zeros(1, self.pose_size).to(self.device)


        self.blink_controller = BlinkController(clock)
        self.mouth_shape_controller = MouthShapeController(clock, None)
        self.body_controller = BodyController(clock)

        self.background_color = background_color

        self.clock = clock

        self.saying_something_ = False
        self.do_blink = True
        self.upscale = upscale
        if upscale:
            self.upscaler = upscale_method()

        self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike("neutral_face.png")).to(self.device)

    def saying_something(self):
        return self.saying_something_

    def set_mouth_shape_sequenece(self, mouth_shape_sequence, emotion_label):
        self.saying_something_ = True
        self.mouth_shape_controller = MouthShapeController(self.clock, mouth_shape_sequence)
        print(emotion_label)
        if emotion_label == "happy":
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 1.0
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 1.0
            self.do_blink = False
        elif emotion_label is None:
            return
        else:
            if emotion_label == "awate":
                self.do_blink = False
                self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 0.0
                self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 0.0
            self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(f"{emotion_label}_face.png")).to(self.device)

    def generate(self):
        if self.do_blink:
            blink_rate = self.blink_controller.blink_rate()
            self.pose[0, self.pose_parameters.get_parameter_index("eye_wink_right")] = blink_rate
            self.pose[0, self.pose_parameters.get_parameter_index("eye_wink_left")] = blink_rate

        self.body_controller.control(self.pose, self.pose_parameters)
        finished = self.mouth_shape_controller.control(self.pose, self.pose_parameters)
        if finished:
            self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(f"neutral_face.png")).to(self.device)
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 0.0
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 0.0

            self.do_blink = True
            self.saying_something_ = False

            for mouth_shape in mouth_shapes:
                self.pose[0, self.pose_parameters.get_parameter_index(mouth_shape)] = 0.0


        s = time.time()
        # s = time.time()
        output_image = self.poser.pose(self.torch_input_image, self.pose)[0]
        # print(time.time() - s)
        # import IPython; IPython.embed()
        # print(1 / (time.time() - s))

        s = time.time()
        output_image = output_image.detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA')
        background = PIL.Image.new('RGBA', pil_image.size, self.background_color)
        pil_image = PIL.Image.alpha_composite(background, pil_image).convert('RGB')
        # print(1 / (time.time() - s))

        if self.upscale:
            # pil_image = pil_image.resize((pil_image.width * 4, pil_image.height * 4), PIL.Image.ANTIALIAS)
            s = time.time()
            pil_image = self.upscaler.upscale(pil_image)
            print("Real ESRGan", 1 / (time.time() - s))
            # output_image_2 *= 255
            # output_image_2 = output_image_2[:,:,::-1]

            # pil_image.show()
            

        return pil_image

import cv2
import sentencepiece as spm

class WallClock():
    def time(self):
        return time.time()

class Speaker():
    def output(self, wav, fs):
        play_obj = sa.play_buffer(wav, 1, 2, fs)

from english_to_kana import EnglishToKana

class SentimentJaFeelingEstimator:
    def __init__(self):
        self.emotion_analyzer = onnxruntime.InferenceSession(
            get_model_file_from_gdrive("sentiment.onnx", "https://drive.google.com/uc?id=1ij9WEObAUJir60qpR1RERlB4-ewiPFVZ"), 
            providers = ['CPUExecutionProvider'])
        
        self._sp = spm.SentencePieceProcessor()
        self._sp.load("sp.model")
        self._maxlen = 281
        self.emotion_label = [
            "happy", "sad", "angry", "disgust", "surprise", "fear"
        ]
    def get_feeling(self, sentence):
        word_ids = self._sp.EncodeAsIds(sentence)
        padded = np.pad(word_ids, (self._maxlen - len(word_ids), 0), 'constant', constant_values=(0, 0))
        emotions = self.emotion_analyzer.run([], {"embedding_input": [padded]})[0][0]
        emotion_index = np.argmax(emotions)
        emotion_label = self.emotion_label[emotion_index] if emotions[emotion_index] > 0.9 else None

        print(emotions)

        return emotion_label


from simpletransformers.classification import ClassificationModel, ClassificationArgs

class FeelingJaFeelingEstimator:
    CLASS_NAMES = [
        'angry_face', 'crying_face', 'face_with_crossed-out_eyes', 'face_with_open_mouth', 
        'flushed_face', 'grinning_face_with_smiling_eyes', 'loudly_crying_face', 'pouting_face', 
        'slightly_smiling_face', 'smiling_face_with_smiling_eyes', 'sparkles', 'tired_face']

    feeling_face_map = {
        'angry_face': 'angry',
        'crying_face': 'sad',
        "face_with_crossed-out_eyes": "awate",
        "face_with_open_mouth": "neutral",
        "flushed_face": "embarrassed",
        "grinning_face_with_smiling_eyes": "happy",
        "loudly_crying_face": "cry",
        "pouting_face": "angry",
        "slightly_smiling_face": "neutral",
        "smiling_face_with_smiling_eyes": "happy",
        "sparkles": "sparkles",
        "tired_face": "awate"
    }

    def __init__(self):
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id="xiongjie/face-expression-ja")
        model_args = ClassificationArgs()
        model_args.onnx = True
        self.model_onnx = ClassificationModel(
            'auto',
            path,
            use_cuda=False,
            args=model_args
        )

    def get_feeling(self, sentence):
        class_id = self.model_onnx.predict([sentence])[0][0]
        return self.feeling_face_map[self.CLASS_NAMES[class_id]]

class TsukuyomichanVisualizer:
    # load the example file included in the ESPnet repository
    MAX_WAV_VALUE = 32768.0
    fs = 24000

    def __init__(self, 
        talksoft, clock=WallClock(), wav_output=Speaker(), background_color=None,
        feeling_estimator=FeelingJaFeelingEstimator
    ):
        d = ModelDownloader()
        aaa = d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave")
        # sentence = "今日の天気はとても良い。"
        sentence = "明日は散歩にでかけたいですね。"
        self.talksoft = talksoft

        self.aligner = CTCSegmentation( **aaa , fs=self.fs )

        self.aligner.set_config( gratis_blank=True, kaldi_style_text=False )
        
        self.nlp = spacy.load('ja_ginza_electra')

        self.clock = clock
        self.wav_output = wav_output

        self.visualization_generator = TsukuyomichanVisualizationGenerator(self.clock, background_color)
        self.feeling_estimator = feeling_estimator()
        base_dictionary = {
            "github": "ギットハブ",
            "FastSRGAN": "ファストエスアールガン",
            "GAN": "ガン",
            "ESRGAN": "イーエスアールガン",
            "Real": "リアル",
            "Bicubic": "バイキュービック",
            "Realtime": "リアルタイム",
            "GB": "ギガバイト",
            "ORT": "オーアールティー",
            "1GB": "イチギガバイト",
            "3D": "スリーディー",
            "Live2D": "ライブツーディー"
        }

        self.english_to_kana_dictionary = {}
        # TODO: decide it is necessary.
        for key in base_dictionary.keys():
            self.english_to_kana_dictionary[key.lower()] = base_dictionary[key]
            self.english_to_kana_dictionary[key] = base_dictionary[key]

        self.english2kana = EnglishToKana()

    def saying_something(self):
        return self.visualization_generator.saying_something()

    def visualize(self, sentence):
        # print("wait visualization")
        if sentence is not None:
            print(f"got {sentence}")
            
            sentence = ''.join(['' if c in emoji.UNICODE_EMOJI['en'] else c for c in sentence])
            sentence = re.sub("<unk>", "", sentence)
            sentence = re.sub(r":.+", "", sentence)
            sentence = sentence.strip()

            emotion_label = self.feeling_estimator.get_feeling(sentence )

            alphabet_replaced_sentence = ""
            doc = self.nlp(sentence)
            for sent in doc.sents:
                for token in sent:
                    if re.match(r"[a-zA-Z]+", token.orth_):
                        yomi = ginza.reading_form(token)
                        if re.match(r"[a-zA-Z]+", yomi) and yomi.lower() in self.english_to_kana_dictionary:
                            yomi = self.english_to_kana_dictionary[yomi.lower()]
                        another_yomi = self.english2kana.convert(yomi.lower())
                        if re.match(r"[a-zA-Z]+", yomi) and another_yomi is not None:
                            yomi = another_yomi
                        # import IPython; IPython.embed()
                        alphabet_replaced_sentence += yomi
                    else:
                        alphabet_replaced_sentence += token.orth_

            print(alphabet_replaced_sentence)

            sentence = alphabet_replaced_sentence

            uttrs = []
            readings = []
            types = []
            doc = self.nlp(sentence)
            for sent in doc.sents:
                for token in sent:
                    if token.pos_ != 'PUNCT' and token.pos_ != 'SYM' and token.pos_ != "X":
                        if token.orth_.strip() !=  "":
                            uttrs.append(token.orth_)
                            readings.append(ginza.reading_form(token))
                            types.append(token.pos_)

            sentence = "".join(uttrs)
            s = time.time()
            print(f"audio is generated from {alphabet_replaced_sentence}")
            if len(alphabet_replaced_sentence) > 200:
                all_wavs = []
                for partial_sentence in alphabet_replaced_sentence.split("。"):
                    wav = self.talksoft.generate_voice(partial_sentence, 1)
                    wav = wav * self.MAX_WAV_VALUE
                    wav = wav.astype(np.int16)
                    all_wavs.append(wav)
                wav = np.concatenate(wav)
                print(time.time() - s)
            else:
                wav = self.talksoft.generate_voice(alphabet_replaced_sentence, 1)
                wav = wav * self.MAX_WAV_VALUE
                wav = wav.astype(np.int16)
                print(time.time() - s)


            for uttr, reading in zip(uttrs, readings):
                print(uttr, reading)

            segments = None
            try:
                segments = self.aligner(wav, uttrs)
            except IndexError:
                import traceback
                traceback.print_exc()
            if segments:
                time_mouth_map = []

                for segment, reading in zip(segments.segments, readings):
                    position = segment[0]
                    div = (segment[1] - segment[0]) / len(reading)
                    for char in reading:
                        mouth_shape = None
                        for mouth_item in mouth_map:
                            if char in mouth_item[0]:
                                mouth_shape = mouth_item[1]
                                break

                        time_mouth_map.append(
                            [position, position+div, mouth_shape]
                        )
                        position += div

                print(time_mouth_map)
                if time_mouth_map is not None:
                    self.visualization_generator.set_mouth_shape_sequenece(
                        time_mouth_map, emotion_label
                    )
                    time_mouth_map = None
                
            self.wav_output.output(wav, self.fs)

        pil_image = self.visualization_generator.generate()

        return pil_image
        # pil_image = pil_image.resize((pil_image.width // 2, pil_image.height // 2), PIL.Image.ANTIALIAS)
        # import io
        # s = time.time()
        # bio = io.BytesIO()
        # pil_image.save(bio, format="PNG")
        # del next_animation_image
    
        
        # window["tsukuyomi_image"].update(bio.getvalue())
        # print(time.time() - s)


        # print("outputting image")

        # image_queue.put_nowait(pil_image)
        # print(time.time() - s)
    # cv2.imshow("tsukuyomi chan", np.array(pil_image)[:, :, ::-1])
    # # cv2.imshow("tsukuyomi chan", np.array(pil_image)[:, :, ::-1])
    # cv2.waitKey(3)
