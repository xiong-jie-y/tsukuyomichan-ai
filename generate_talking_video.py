import os
import numpy as np
from tsukuyomichan_talksoft import TsukuyomichanTalksoft
from agent_display import TsukuyomichanVisualizer

def generate_video(output_path, text, subtitle_path=None):
    import re

    manuscript = [s for s in text.split("\n") if s.strip() != ""]
    for i in range(0, len(manuscript)):
        if re.match(r"[0-9\.]+", manuscript[i]):
            manuscript[i] = float(manuscript[i])

    import tempfile
    from moviepy.editor import AudioFileClip, CompositeAudioClip
    from moviepy.audio.AudioClip import AudioArrayClip
    from moviepy.video.io import ImageSequenceClip

    class VideoGenerator:
        def __init__(self, d_path):
            self.frame = 0
            self.frame_rate = 30
            self.pil_images = []
            self.wav = np.array([], dtype=np.int16)
            self.fs = 24000
            self.subtitles = []
            self.d_path = d_path
            self.image_files = []

        def time(self):
            return self.frame / self.frame_rate

        def output(self, wav, fs):
            self.wav = np.concatenate((self.wav, wav))

        def pad_with_empty(self):
            subtract_wavlen_videolen = int(self.fs * (self.time() - len(self.wav)/self.fs))
            if subtract_wavlen_videolen > 0:
                self.wav = np.concatenate((self.wav, np.array([0] * subtract_wavlen_videolen)))
                self._save_current_pil_images()
                return True
            elif subtract_wavlen_videolen == 0:
                self._save_current_pil_images()
                return  True
            else:
                return False

        def add_video_frame(self, pil_image):
            self.pil_images.append(pil_image)
            self.frame += 1

        def add_subtitle(self, start_time, end_time, text):
            self.subtitles.append({
                "start_time": start_time,
                "end_time": end_time,
                "text": text
            })

        def _save_current_pil_images(self):
            for i, pil_image in enumerate(self.pil_images):
                filepath = os.path.join(self.d_path, f"frame_{len(self.image_files)}.png")
                pil_image.save(filepath)
                self.image_files.append(filepath)
            self.pil_images.clear()

        def output_video(self):
            # import IPython; IPython.embed()
            self.wav = self.wav.astype(np.int16)
            from scipy.io.wavfile import write
            write(os.path.join(self.d_path, "tmp.wav"), self.fs, self.wav)
            self._save_current_pil_images()
            # audio = AudioArrayClip([self.wav], fps=self.fs)
            audio = AudioFileClip(os.path.join(self.d_path, "tmp.wav"))
            # new_audioclip = CompositeAudioClip([audio])
            clip = ImageSequenceClip.ImageSequenceClip(self.image_files, fps=self.frame_rate)
            clip.audio = audio
            clip.write_videofile(output_path)
            # import IPython; IPython.embed()

            def second_to_string(second):
                hour = second // 3600
                remain = second % 3600
                minute = remain // 60
                remain = remain % 60
                second = remain // 1
                millisecond = remain % 1
                return "%02d:%02d:%02d,%03d" % (hour, minute, second, millisecond * 1000)

            if subtitle_path is not None:
                text = ""
                for i, subtitle in enumerate(self.subtitles):
                    text += f"{i}\n"
                    text += second_to_string(subtitle["start_time"]) + " --> " + second_to_string(subtitle["end_time"]) + "\n"
                    text += subtitle["text"] + "\n\n"

                open(subtitle_path, "w").write(text)
                

    with tempfile.TemporaryDirectory() as d_path:
        video_generator = VideoGenerator(d_path)
        visualizer = TsukuyomichanVisualizer(
            TsukuyomichanTalksoft(model_version='v.1.2.0'), clock=video_generator, wav_output=video_generator, background_color=(0,255,0))

        # import IPython; IPython.embed()
        for text in manuscript:
            if isinstance(text, str):
                while not video_generator.pad_with_empty():
                    video_generator.add_video_frame(visualizer.visualize(None))

                # Starting subtitle.
                subtitle_start_time = video_generator.time()

                video_generator.add_video_frame(visualizer.visualize(text))

                while visualizer.saying_something():
                    video_generator.add_video_frame(visualizer.visualize(None))

                subtitle_end_time = video_generator.time()
                video_generator.add_subtitle(subtitle_start_time, subtitle_end_time, text)
            elif isinstance(text, float):
                next_clock = video_generator.time() + text
                i = 0
                while video_generator.time() < next_clock:
                    video_generator.add_video_frame(visualizer.visualize(None))
                    i+= 1
                print(i)

        # This is necessary to pad last phrase.
        while not video_generator.pad_with_empty():
            video_generator.add_video_frame(visualizer.visualize(None))


        video_generator.output_video()
