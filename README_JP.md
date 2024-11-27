**要約**: 私は非常に素晴らしい日本語音声生成モデルを作成しました。

[個人プロジェクト](https://github.com/Respaired/Project-Kanade)の一部で、日本語Speech分野のさらなる発展に焦点を当てています。 

- **Tsukasa** (24kHz)のHuggingFaceスペースを使用してください: [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Shiki)
- **Tsumugi** (48kHz)のHuggingFaceスペース: [![huggingface](https://img.shields.io/badge/Interactive_Demo-HuggingFace-yellow)](https://huggingface.co/spaces/Respair/Shiki)

- Shoukan labのDiscordサーバーに参加してください、私がよく訪れる居心地の良い場所です -> [![Discord](https://img.shields.io/discord/1197679063150637117?logo=discord&logoColor=white&label=Join%20our%20Community)](https://discord.gg/JrPSzdcM)

## これは何?

*注意*: このモデルは日本語のみをサポートしていますが、Gradioデモでローマ字、またローマ字と普通の日本語をミックスしたテキストを入力することができます。

これはスピーチ生成ネットワークで、生成された音声の表現力と制御性を最大化することを目的としています。その中核にあるのは[StyleTTS 2](https://github.com/yl4579/StyleTTS2)のアーキテクチャで、以下のような変更が加えられています:

- 完全に新しいデータ前処理パイプラインの導入
- 通常のPyTorch LSTMレイヤーではなくmLSTMレイヤーを採用し、テキストおよびプロソディエンコーダーの容量を高めるためにパラメーター数を増やしている
- PL-Bert、Pitch Extractor、Text Alignerを一から再学習
- SLMにはWavLMの代わりにWhisperのエンコーダーを使用
- 48kHzの設定
- 非言語サウンド(溜息、ポーズなど)や笑い声の表現力が向上
- スタイルベクトルのサンプリングの新しい方法
- プロンプト可能な音声合成
- ローマ字の入力や日本語とローマ字の混在に対応した賢いフォネミゼーションアルゴリズム
- DDP(Distributed Data Parallel)とBF16(Bfloat16)の訓練が修正された(ほとんど!)

2つのチェックポイントが使用できます。Tsukasa and Tsumugi(仮称)です。

Tsukasaは約800時間のデータで学習されています。主にゲームやノベルからのデータで、一部はプライベートデータセットからのものです。
そのため、日本語は「アニメ日本語」(実際の日常会話とは異なる)になります。

Tsumugi(仮称)は、この データの一部約300時間を使用し、さらに手動クリーニングや注釈付けを行った制御された方法で学習されています。 

残念ながら、Tsumugiのコンテキスト長は制限されているため、イントネーションの処理はTsukasaほど良くありません。
また、Kotodamaのインファレンスの最初のモードしかサポートしていないため、ボイスデザインはできません。


提供:

- Soshyant (私)
- Auto Meta (Alignment AI Lab)
- Cryptowooser
- Buttercream

## なぜ重要なのですか?

最近、より大規模なモデルへの傾向がありますが、私は逆の道を行き、既存のツールを活用することで限界まで性能を引き上げることを試みています。
スケールが高くなくてもいい結果が得られるかもしれないことを試しています。

日本語に関連するいくつかの事項もあります。例えば、この言語のイントネーションをどのように改善できるか、文脈によって綴りが変わる文章をどのように正確に注釈付けできるかなどです。

## 使い方

# Inference:

Gradioデモ:
```bash
python app_tsuka.py
```

または、推論ノートブックをチェックしてください。その前に、**重要な注意事項**セクションをよく読んでください。

# Training:

第1段階:
```bash
accelerate launch train_first.py --config_path ./Configs/config.yml
```
第2段階 **(DDPバージョンが動作しないため、現在のバージョンではDPを使用しています。[#7](https://github.com/yl4579/StyleTTS2/issues/7)を参照して、ヘルプをお願いします)**:
```bash
accelerate launch accelerate_train_second.py --config_path ./Configs/config.yml 
```

SLMの共同TrainはマルチGPUでは機能しません。(そもそもこの段階を行うのは必要かどうか自体が疑問です、私も使用していません。)

または:

```bash
launch train_first.py --config_path ./Configs/config.yml
```

第3段階(Kotodama、プロンプトエンコーディングなど):
*未予定*


## 今後の改善案

いくつかの改善点が考えられます。必ずしも私が取り組むわけではありませんが、提案として捉えてください:

- [o] デコーダーの変更(具体的に[fregrad](https://github.com/kaistmm/fregrad)が面白そう。)
- [o] 別のアルゴリズムを使ってPitch Extractorを再訓練
- [o] 非音声サウンドの生成は改善されましたが、完全な非音声出力は生成できません。これは、hard alignmentの影響かもしれません。
- [o] スタイルエンコーダーを別のモダリティとしてLLMsで使用する(Style-Talkerに似たアプローチ)

## 前提条件
1. Python >= 3.11
2. このリポジトリをクローンします:
```bash
git clone https://github.com/yl4579/StyleTTS2.git
cd StyleTTS2
```
3. Pythonの要件をインストールします: 
```bash
pip install -r requirements.txt
```

## 訓練の詳細

- 8x A40s + 2x V100s(32GBずつ)
- 750 ~ 800時間のデータ
- Bfloat16
- 約3週間の訓練、全体で3ヶ月(データパイプラインの作業を含む)
- Google Cloudベースで概算すると、66.6 kg CO2eq.の二酸化炭素排出(Google Cloudは使用していませんが、クラスターがアメリカにあるため、非常に大まかな推定です。) 


### 重要な注意事項

1. ディフュージョンサンプラーを有効にすると出力が壊れます:

***残念ながら、ハードウェアによっては、ディフュージョンサンプラーが機能しない可能性があります。この問題は私の管理外で、異なるハードウェアが浮動小数点演算を処理する方法に関連しているようです。A40、V100、Google Colabの T4 GPUでは機能することを確認していますが、同じGPUを持っていても動作を保証することはできません。CPUを使用しても同じ問題が起こります。これは元のStyleTTS2で深刻な問題でしたが、私が追加したさまざまなサンプリング方法を使うことで、品質への影響を最小限に抑えつつ、ディフュージョンサンプラーを無効にすることができます。***

2. 提供されたサンプルの品質を再現できない、または一貫して良い結果が得られない:

***可制御性には代償がかかり、それはユーザビリティの低下です。特に、本質的に非決定的なモジュールで構成されたネットワークの場合に当てはまります。システムはスタイルベクトルの変動に非常に敏感です。ただし、推論パラメーターを慎重に調整し、試行錯誤すれば、ほとんど常に最も印象的な自然な表現を達成できると確信しています。また、一部のスピーカーは特定の感情を一貫して処理できない可能性があるため、別のスピーカーから新しい感情を作り出すことができます。Gradioスペースや推論ノートブックでの詳しい使用方法を説明しています。***

3. [RuntimeError: The size of tensor a (512) must match the size of tensor b (some number) at non-singleton dimension 3]:

***入力が1回の推論に対して長すぎます。Longform推論機能を使用してください。これは特に、Tsumugi(仮称)チェックポイントでは問題になります。mLSTMレイヤーのコンテキスト長が512に制限されているため、Longform機能を使用しない限り、約10秒以上の音声を生成できません。ただし、他のチェックポイントではこれは問題にはなりません。Longform アルゴリズムのおかげで、出力の長さに理論的な制限はありません。***

3. 短い入力が印象的ではない:

***2で述べたことがすべて当てはまります。スタイルベクトルが適切かどうかを確認してください。ただし、一般的に非常に短い入力の使用は推奨されません。***

4. 2段階目の訓練でNaNが発生:

***グラジエントが爆発しているのかもしれません。クリッピングを試すか、バッチサイズが大すぎる可能性があります。それでも解決しない場合は、オリジナルのDPスクリプトを使って最初の数エポックを事前訓練することをお勧めします。または、完全にDPを使用してください。***