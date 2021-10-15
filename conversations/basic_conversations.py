import time

from bert_agent import EmbeddingBasedReplyAgent
from yamnet import SoundEventType

class CommandConversation:
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

class SighCaringConversation:
    def __init__(self, agent):
        pass

    def react_to(self, event):
        return event.type == SoundEventType.TooManySigh

    def start(self, event, agent):
        agent.speak("ため息が多いけど、大丈夫？", seed=10)

        user_reply = agent.wait_for_one_of_in_similar_meaning(
            ["はい", "いいえ"],
            timeout=5
        )

        if user_reply == "はい":
            agent.speak("無理しないでね!")
        elif user_reply == "いいえ":
            agent.speak("大丈夫？病院行ったほうがいいよ。")
        else:
            agent.speak("頑張ってね!")