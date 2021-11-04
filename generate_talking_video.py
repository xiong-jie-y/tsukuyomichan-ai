import os
import numpy as np
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
from agent_display import TsukuyomichanVisualizer


# manuscript = [
#     "こんにちは、フリー素材キャラクターつくよみの非公式AIです。",
#     0.8,
#     "私、最近、文章読み動画を作ることができるようになりました。",
#     0.8,
#     "文章を用意していただければ、読み上げて動画にします。",
#     0.3,
#     "読み上げ内容に合わせて、口を動かしたり、表情を変えたりもできます。",
#     0.5, 
#     "例えば",
#     0.5, 
#     "むかつく",
#     0.3, 
#     "というように怒りの表現であれば、怒った表情になり",
#     0.3,
#     "今日のカレーは美味しかった",
#     0.3,
#     "というようにハッピーな表現であれば、ハッピーな表情になります。",
#     0.3,
#     "この動画も文章を元に自動的に私が作りました。",
#     1.0
# ]

manuscript = """
Talking Head Anime 2とは、一枚のキャラクターの顔画像のみを用いてキャラクターの口や目や眉毛を動かせる技術です。
Githubでオープンソースで公開されています。
"""

manuscript = """
これは、超解像モデルの中でもそこそこ動く、Real ESRGAN、FastSRGAN、Realtime Super ResolutionをBicubic補完を使った拡大結果です。
実行速度はRTX 3070のPCで測定しました。
アニメ画像の場合、Real ESRGANは非常にきれいに拡大してくれました。
実行速度は10fps前後で、キャラクターのモーションをリアルタイムで表示するには厳しく、動画生成など、非リアルタイム処理に用いました。
FastSRGANはBicubicよりましですが、若干ぼやけています。
ただ、30fpsで動かすことができるため、512×512に縮小してリアルタイム処理に利用しました。
Realtime Super Resolutionは実行速度が遅い割には、そんなにきれいではなかったため、今回は使いませんでした。
1.5
また、さらなる高速化を目指し、ONNX化とgraph optimizationも試してみました。
ONNX化したモデルでは、消費ビデオメモリ量が数百から1GB増え、実行速度はあまり変わりませんでした。
ORT化などさらなる最適化をかけた場合は実行速度が低下しました。
これ以上の速度改善には今の所成功していません。
"""

import re

manuscript = [s for s in manuscript.split("\n") if s.strip() != ""]
for i in range(0, len(manuscript)):
    if re.match(r"[0-9\.]+", manuscript[i]):
        manuscript[i] = float(manuscript[i])

import tempfile
from moviepy.editor import AudioFileClip, CompositeAudioClip
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io import ImageSequenceClip

class VideoGenerator:
    def __init__(self):
        self.frame = 0
        self.frame_rate = 30
        self.pil_images = []
        self.wav = np.array([], dtype=np.int16)
        self.fs = 24000

    def time(self):
        return self.frame / self.frame_rate

    def output(self, wav, fs):
        self.wav = np.concatenate((self.wav, wav))

    def pad_with_empty(self):
        subtract_wavlen_videolen = int(self.fs * (self.time() - len(self.wav)/self.fs))
        if subtract_wavlen_videolen > 0:
            self.wav = np.concatenate((self.wav, np.array([0] * subtract_wavlen_videolen)))
            return True
        elif subtract_wavlen_videolen == 0:
            return  True
        else:
            return False

    def add_video_frame(self, pil_image):
        self.pil_images.append(pil_image)
        self.frame += 1

    def output_video(self):
        # import IPython; IPython.embed()
        image_files = []
        self.wav = self.wav.astype(np.int16)
        from scipy.io.wavfile import write
        with tempfile.TemporaryDirectory() as d_path:
            write(os.path.join(d_path, "tmp.wav"), self.fs, self.wav)
            for i, pil_image in enumerate(self.pil_images):
                filepath = os.path.join(d_path, f"frame_{i}.png")
                pil_image.save(filepath)
                image_files.append(filepath)
            # audio = AudioArrayClip([self.wav], fps=self.fs)
            audio = AudioFileClip(os.path.join(d_path, "tmp.wav"))
            # new_audioclip = CompositeAudioClip([audio])
            clip = ImageSequenceClip.ImageSequenceClip(image_files, fps=self.frame_rate)
            clip.audio = audio
            clip.write_videofile('my_video.mp4')
            # import IPython; IPython.embed()
            
video_generator = VideoGenerator()
visualizer = TsukuyomichanVisualizer(
    TsukuyomichanTalksoft(model_version='v.1.2.0'), clock=video_generator, wav_output=video_generator, background_color=(0,255,0))

# import IPython; IPython.embed()
for text in manuscript:
    if isinstance(text, str):
        while not video_generator.pad_with_empty():
            video_generator.add_video_frame(visualizer.visualize(None))
        video_generator.add_video_frame(visualizer.visualize(text))

        while visualizer.saying_something():
            video_generator.add_video_frame(visualizer.visualize(None))
    elif isinstance(text, float):
        next_clock = video_generator.time() + text
        i = 0
        while video_generator.time() < next_clock:
            video_generator.add_video_frame(visualizer.visualize(None))
            i+= 1
        print(i)

video_generator.output_video()
