## つくよみちゃん会話AI
これは、つくよみちゃん会話AIです。
AlexaやGoogleHomeのAIアシスタントのように
１往復〜３往復くらいの会話が音声ベースでできる設計のシステムです。（現在１往復のみ）

将来的にはアシスタント機能が実装されたり、
周辺状況に応じた発言機能なども実装される予定です。

## インストール
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## 使い方

次のコマンドでAIを起動してください。
```
python listen_user.py
```

ターミナルに「Silence」という文字が表示されたら、「ねぇ」とマイクに話しかけてください。

その後、「明日の天気は？」と聞いてみてください。
返事が可愛い声で返ってきます。

## 利用規約
[利用規約とクレジット](LICENSE.md)