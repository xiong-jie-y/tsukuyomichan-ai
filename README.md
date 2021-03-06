## つくよみちゃん会話AI
これは、つくよみちゃん会話AIです。
AlexaやGoogleHomeのAIアシスタントのように
１往復〜３往復くらいの会話が音声ベースでできる設計のシステムです。（現在１往復のみ）

将来的にはアシスタント機能が実装されたり、
周辺状況に応じた発言機能なども実装される予定です。

## インストール
```
# OSがUbuntuの場合のみ
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## 使い方

次のコマンドでAIを起動してください。
```
python listen_user.py
```

ターミナルに「Silence」という文字が表示されたら、「ねぇ」とマイクに話しかけてください。

その後、明日の天気を聞いてみてください。
返事が可愛い声で返ってきます。（実際の天気予報の情報ではありません。）

## 会話の追加の仕方
`conversations/basic_conversations.py`ファイル内会話の処理が書かれているので、
ここに新しいクラスを追加することで、新しい会話処理を追加できます。
下記は、ため息を吐きすぎた場合に発生する会話処理の例です。

```python
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
```

## 利用規約
[利用規約とクレジット](LICENSE.md)