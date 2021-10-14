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
このリポジトリ内のコードはMITライセンスです。

音声合成に[tsukuyomichan-talksoft](https://github.com/shirowanisan/tsukuyomichan-talksoft)を使っています。tsukuyomichan-talksoftで生成された音声に関しては、[tsukuyomichan-talksoft](https://github.com/shirowanisan/tsukuyomichan-talksoft)の[利用規約](https://shirowanisan.com/tyc-talksoft-terms)に従ってください。

利用規約によると、つくよみちゃんにどのような発言をさせてもいいわけではありません。一例として、人を批判したり攻撃する音声を生成してはいけないという条項があります。

## クレジット
本コンテンツは「シロワニさんのつくよみちゃんトークソフト」の音声合成モデルを使用しています。

■シロワニさんのつくよみちゃんトークソフト

https://shirowanisan.com/tyc-talksoft

© shirowanisan

「シロワニさんのつくよみちゃんトークソフト」の音声合成モデルは、フリー素材キャラクター「つくよみちゃん」が無料公開している音声データから作られています。

■つくよみちゃんコーパス（CV.夢前黎）

https://tyc.rei-yumesaki.net/material/corpus/

© Rei Yumesaki