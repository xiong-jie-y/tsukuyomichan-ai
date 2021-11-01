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

# import simpleaudio as sa
# play_obj = sa.play_buffer(wav, 1, 2, fs)

class BlinkController:
    def __init__(self, start_time):
        self.state_start_time = start_time
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

        duration_from_last = time.time() - self.state_start_time

        # print(self.state)

        if duration_from_last > state_periods[self.state]:
            self.state_start_time = time.time()
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
    def __init__(self, start_time):
        self.state_start_time = start_time
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

        duration_from_last = time.time() - self.state_start_time

        # print(self.state)

        if duration_from_last > state_periods[self.state]:
            self.state_start_time = time.time()
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
        if rate > 0.0:
            pose[0, pose_parameters.get_parameter_index("eye_relaxed_left")] = 1.0
            pose[0, pose_parameters.get_parameter_index("eye_relaxed_right")] = 1.0
            pose[0, pose_parameters.get_parameter_index("eye_wink_left")] = 0.0
            pose[0, pose_parameters.get_parameter_index("eye_wink_right")] = 0.0
        else:
            pose[0, pose_parameters.get_parameter_index("eye_relaxed_left")] = 0
            pose[0, pose_parameters.get_parameter_index("eye_relaxed_right")] = 0

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
    def __init__(self, time_mouth_map):
        self.start_time = time.time()
        self.time_mouth_map = time_mouth_map
        self.discrete_parameter = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def control(self, pose, pose_parameters):
        if self.time_mouth_map is None:
            for mouth_shape in mouth_shapes:
                pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0
            return False

        duration_from_sentence_start = (time.time() - self.start_time)

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
        if name is not None: #  and (time.time() - last_utterance) > 0.05:
            # print(progress_rate_in_utterance)
            print(f"Controlling with {current_time_mouth}: {progress_rate_in_utterance}")
    
            if name != "mouth_nnn":
                pose[0, pose_parameters.get_parameter_index(name)] = progress_rate_in_utterance
            for mouth_shape in mouth_shapes:
                if mouth_shape != name:
                    pose[0, pose_parameters.get_parameter_index(mouth_shape)] = 0.0

        last_utterance = time.time()
        return False

import queue
import simpleaudio as sa
import re

import onnxruntime
            
# from pyanime4k import ac

class TsukuyomichanVisualizationGenerator:
    def __init__(self, upscale=True):
        self.device = torch.device("cuda:0")

        self.poser = tha2.poser.modes.mode_20.create_poser(self.device)
        self.pose_parameters = tha2.poser.modes.mode_20.get_pose_parameters()
        self.pose_size = self.poser.get_num_parameters()
        self.pose = torch.zeros(1, self.pose_size).to(self.device)


        self.blink_controller = BlinkController(time.time())
        self.mouth_shape_controller = MouthShapeController(None)
        self.body_controller = BodyController(time.time())

        self.do_blink = True
        self.upscale = upscale
        if upscale:
            self.super_resolution_session = onnxruntime.InferenceSession("generator3.onnx", providers = ['CUDAExecutionProvider'])

        self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike("neautral_face.png")).to(self.device)

    def set_mouth_shape_sequenece(self, mouth_shape_sequence, emotion_label):
        self.mouth_shape_controller = MouthShapeController(mouth_shape_sequence)
        print(emotion_label)
        if emotion_label == "happy":
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 1.0
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 1.0
            self.do_blink = False
        if emotion_label == "sad" or emotion_label == "angry":
            self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(f"{emotion_label}_face.png")).to(self.device)

    def generate(self):
        if self.do_blink:
            blink_rate = self.blink_controller.blink_rate()
            self.pose[0, self.pose_parameters.get_parameter_index("eye_wink_right")] = blink_rate
            self.pose[0, self.pose_parameters.get_parameter_index("eye_wink_left")] = blink_rate

        # self.body_controller.control(self.pose, self.pose_parameters)
        finished = self.mouth_shape_controller.control(self.pose, self.pose_parameters)
        if finished:
            self.torch_input_image = extract_pytorch_image_from_PIL_image(extract_PIL_image_from_filelike(f"neautral_face.png")).to(self.device)
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_right")] = 0.0
            self.pose[0, self.pose_parameters.get_parameter_index("eye_happy_wink_left")] = 0.0

            self.do_blink = True

            for mouth_shape in mouth_shapes:
                self.pose[0, self.pose_parameters.get_parameter_index(mouth_shape)] = 0.0


        s = time.time()
        # s = time.time()
        output_image = self.poser.pose(self.torch_input_image, self.pose)[0]
        # print(time.time() - s)
        # import IPython; IPython.embed()
        print(1 / (time.time() - s))

        s = time.time()
        output_image = output_image.detach().cpu()
        numpy_image = numpy.uint8(numpy.rint(convert_output_image_from_torch_to_numpy(output_image) * 255.0))
        pil_image = PIL.Image.fromarray(numpy_image, mode='RGBA').convert('RGB')
        print(1 / (time.time() - s))

        if self.upscale:
            # pil_image = pil_image.resize((pil_image.width * 4, pil_image.height * 4), PIL.Image.ANTIALIAS)
            s = time.time()
            # import IPython; IPython.embed()
            output_image_2 = self.super_resolution_session.run([], {"input_1": [np.array(pil_image)/255]})[0][0]
            # output_image_2 *= 255
            # output_image_2 = output_image_2[:,:,::-1]

            output_image_2 = (((output_image_2 + 1) / 2.) * 255).astype(np.uint8)
            # import IPython; IPython.embed()
            print(1 / (time.time() - s))
            pil_image = PIL.Image.fromarray(numpy.uint8(numpy.rint(output_image_2.reshape(1024, 1024, 3))), mode='RGB')
            pil_image = pil_image.resize((pil_image.width // 2, pil_image.height //2 ))
            # pil_image.show()
            

        return pil_image

import cv2
import sentencepiece as spm

class TsukuyomichanVisualizer:
    # load the example file included in the ESPnet repository
    MAX_WAV_VALUE = 32768.0
    fs = 24000

    def __init__(self, talksoft):
        d = ModelDownloader()
        aaa = d.download_and_unpack("kan-bayashi/csj_asr_train_asr_transformer_raw_char_sp_valid.acc.ave")
        # sentence = "今日の天気はとても良い。"
        sentence = "明日は散歩にでかけたいですね。"
        self.talksoft = talksoft

        self.aligner = CTCSegmentation( **aaa , fs=self.fs )

        self.aligner.set_config( gratis_blank=True, kaldi_style_text=False )
        
        self.nlp = spacy.load('ja_ginza_electra')

        self.visualization_generator = TsukuyomichanVisualizationGenerator()
        self.emotion_analyzer = onnxruntime.InferenceSession("sentiment.onnx", providers = ['CPUExecutionProvider'])
        
        self._sp = spm.SentencePieceProcessor()
        self._sp.load("sp.model")
        self._maxlen = 281
        self.emotion_label = [
            "happy", "sad", "angry", "disgust", "surprise", "fear"
        ]

    def visualize(self, visualize_queue, image_queue, window):
        while True:
            # print("wait visualization")
            sentence = None
            try:
                sentence = visualize_queue.get_nowait()
            except queue.Empty:
                pass
            if sentence is not None:
                print(f"got {sentence}")
                
                sentence = ''.join(['' if c in emoji.UNICODE_EMOJI['en'] else c for c in sentence])
                sentence = re.sub("<unk>", "", sentence)
                sentence = re.sub(r":.+", "", sentence)
                sentence = sentence.strip()

                word_ids = self._sp.EncodeAsIds(sentence)
                padded = np.pad(word_ids, (self._maxlen - len(word_ids), 0), 'constant', constant_values=(0, 0))
                emotions = self.emotion_analyzer.run([], {"embedding_input": [padded]})[0][0]
                emotion_index = np.argmax(emotions)
                emotion_label = self.emotion_label[emotion_index] if emotions[emotion_index] > 0.9 else None
                # import IPython; IPython.embed()

                print(emotions)

                s = time.time()
                if len(sentence) > 200:
                    all_wavs = []
                    for partial_sentence in sentence.split("。"):
                        wav = self.talksoft.generate_voice(partial_sentence, 1)
                        wav = wav * self.MAX_WAV_VALUE
                        wav = wav.astype(np.int16)
                        all_wavs.append(wav)
                    wav = np.concatenate(wav)
                    print(time.time() - s)
                else:
                    wav = self.talksoft.generate_voice(sentence, 1)
                    wav = wav * self.MAX_WAV_VALUE
                    wav = wav.astype(np.int16)
                    print(time.time() - s)

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

                print(uttrs, readings, types)
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

                play_obj = sa.play_buffer(wav, 1, 2, self.fs)

            pil_image = self.visualization_generator.generate()
            # pil_image = pil_image.resize((pil_image.width // 2, pil_image.height // 2), PIL.Image.ANTIALIAS)
            # import io
            # s = time.time()
            # bio = io.BytesIO()
            # pil_image.save(bio, format="PNG")
            # del next_animation_image
        
            
            # window["tsukuyomi_image"].update(bio.getvalue())
            # print(time.time() - s)
            cv2.imshow("test", np.array(pil_image)[:,:,::-1])
            cv2.waitKey(3)

            # print("outputting image")

            # image_queue.put_nowait(pil_image)
            # print(time.time() - s)
        # cv2.imshow("tsukuyomi chan", np.array(pil_image)[:, :, ::-1])
        # # cv2.imshow("tsukuyomi chan", np.array(pil_image)[:, :, ::-1])
        # cv2.waitKey(3)
