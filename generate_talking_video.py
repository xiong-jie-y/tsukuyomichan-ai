import os
import numpy as np
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
from agent_display import TsukuyomichanVisualizer

def generate_video(output_path, text):
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
                clip.write_videofile(output_path)
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
