import time

from bert_agent import EmbeddingBasedReplyAgent
from yamnet import SoundEventType

from abc import ABC

class Conversation(ABC):
    def react_to(self, event):
        return False

    def fire(self):
        return False

class CommandConversation(Conversation):
    def __init__(self, agent):
        self.reply_agent = EmbeddingBasedReplyAgent(
            "conversation_dir", agent.sentence_transformer)
        if not self.reply_agent.is_ready():
            self.reply_agent.ingest_csv("conversation_pair.txt")

    def react_to(self, event):
        return event.type == SoundEventType.UserTalked

    def start(self, event, agent):
        text = agent.speech_to_text(event.sound)

        print(f"you said {text}")
        if text not in ["ねぇ", "あ", "つくよみちゃん", "助けて", "困った"]:
            return

        time.sleep(0.3)
        agent.speak("何？")

        sentence = agent.recognize_talk()
        reply = self.reply_agent.reply_to(sentence)
        agent.speak(reply)

        time.sleep(1)

class SighCaringConversation(Conversation):
    def __init__(self, agent):
        pass

    def react_to(self, event):
        return event.type == SoundEventType.TooManySigh

    def start(self, event, agent):
        agent.speak("ため息が多いけど、大丈夫？", seed=10)

        user_reply = agent.wait_for_one_of_in_similar_meaning(
            ["はい", "いいえ"],
            timeout=7
        )

        if user_reply == "はい":
            agent.speak("無理しないでね!")
        elif user_reply == "いいえ":
            agent.speak("大丈夫？病院行ったほうがいいよ。")
        else:
            agent.speak("頑張ってね!")

import datetime

class WakeUserUpTask(Conversation):
    def __init__(self, agent):
        self.last_alarm = datetime.datetime.now()
        self.alarm_time = datetime.time(hour=7, minute=0)

    def fire(self):
        if self.last_alarm.time() < self.alarm_time and \
            self.alarm_time < datetime.datetime.now().time():
            self.last_alarm = datetime.datetime.now()
            return True
        else:
            return False

    def start(self, event, agent):
        retry_count = 0
        while retry_count != 5:
            agent.speak("朝ですよ。起きてください。")

            user_reply = agent.wait_for_one_of_in_similar_meaning(
                ["起きる。", "まだ寝る。"],
                timeout=7
            )

            print(user_reply)

            if user_reply == "起きる。":
                agent.speak("えらいです！")
                return
            elif user_reply == "まだ寝る。":
                agent.speak("こら、早く起きてください。")
            else:
                agent.speak("こらー！！！")

            retry_count += 1
