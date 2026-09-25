# YonmokuRessen Neural Network

[YonmokuRessen](https://github.com/akaken1123/YonmokuRessen)（HP制四目並べ）用の、ニューラルネットワークによる評価・着手選択AIを学習させるためのリポジトリです。Javaの本体アプリとは別リポジトリにして、Python/PyTorchの学習環境を分離しています。

## 全体像

2つの学習方式があります。

**方式A：既存AIの模倣（教師あり学習、`yonmoku_nn.selfplay` / `yonmoku_nn.train`）**
1. YonmokuRessenのJavaサーバーを起動しておき、REST API経由でAI対AI（デフォルトAI・テストAI・テスト3AI・学習AI等）の対局を作らせる。サーバー側が既に持っている `AiVsAiDriver`（対局作成後、人手を介さず自動で最後まで打ち切る仕組み）と `/api/games/{id}/history`（着手ごとの盤面・HP・保留ダメージなどの全履歴）をそのまま使う。
2. 収集した「局面 → 実際に打たれた手 → 最終的な勝敗」のデータから、PyTorchで方策・価値を教師あり学習する。強さは模倣元のAI次第（模倣元より強くはならない）。

**方式B：MCTS自己対戦（強化学習、`yonmoku_nn.rl_selfplay` / `yonmoku_nn.train_rl`）**
1. ネットワーク自身の方策・価値を使ってPUCT（MCTS）で先読みしながら自己対戦する。手番ごとにサーバーの
   ステートレスなシミュレーションAPI（`/api/simulate/move`）を呼んで、実際の対局を作らずに「もしここに
   打ったら」を何度も試す（盤面ルールはここでもJava側の実装をそのまま使う）。
2. 収集した「局面 → MCTSの訪問回数分布（方策ターゲット） → 最終的な勝敗」のデータで学習し直す。
3. 学習したネットワークで再度自己対戦→再学習…を繰り返すことで、模倣元の強さを超えて成長できる
   （AlphaZeroと同じ考え方）。ただし1回のループが方式Aより重い（MCTSのシミュレーション回数分だけ
   サーバーへの通信とネットワーク推論が発生する）。

いずれの方式も、最後にONNX形式へエクスポートする（Java側での推論統合は`YonmokuRessen`本体の
`AiLevel.NEURAL`で対応済み）。対局時もネットワーク単体の貪欲法ではなく、方式Bと同じPUCT探索
（`NeuralMcts`、Java側に移植済み）で先読みしてから手を選ぶ。

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate
pip install -r requirements.txt
```

### GPUについて（Radeon RX 9070 XTなど、AMD GPU利用時の注意）

- `pip install torch` で入る既定のホイールはCPU版です。
- AMD GPUでのアクセラレーションはROCm経由になりますが、ROCmは基本的にLinux（またはWSL2）向けで、特にRDNA4世代（RX 9070 XT等）のように新しいGPUは対応状況が変わりやすいので、実際に使う前に[PyTorch公式のROCm対応状況](https://pytorch.org/get-started/locally/)を確認してください。
- Windowsネイティブで動かしたい場合は `torch-directml` パッケージ（DirectML経由）も選択肢ですが、ROCmネイティブほど枯れていません。
- この盤面サイズ（9×9）・ネットワーク規模であれば、CPUだけでも現実的な時間で学習できます。まずはCPUで動作確認してから、GPU環境を整えるのがおすすめです。

## 使い方

### 1. データ収集（YonmokuRessenサーバーを起動した状態で）

```bash
python -m yonmoku_nn.selfplay --server http://localhost:8080 --games 200 \
  --black-ai TEST3 --white-ai TEST3 --out data/selfplay_test3.jsonl
```

`--black-ai`/`--white-ai` は `DEFAULT`/`TEST`/`TEST2`/`TEST3`/`LEARN` から指定できます（YonmokuRessen側の `AiLevel` と同じ）。強いAI同士の対局を集めるほど、質の良い教師データになります。

### 2. 学習

```bash
python -m yonmoku_nn.train --data "data/*.jsonl" --epochs 20 --out checkpoints/model.pt
```

### 3. ONNXへエクスポート

```bash
python -m yonmoku_nn.export --checkpoint checkpoints/model.pt --out checkpoints/model.onnx
```

### 方式B：MCTS自己対戦（強化学習ループ）

YonmokuRessenサーバーを起動した状態で（対局を作る必要はない。ステートレスなシミュレーションAPIだけを使う）：

```bash
# 1. 自己対戦でデータを集める（--checkpointを省略するとランダム初期化のネットワークから始まる）
python -m yonmoku_nn.rl_selfplay --server http://localhost:8080 \
  --checkpoint checkpoints/model.pt --games 20 --simulations 100 \
  --out data/rl_selfplay_gen1.jsonl

# 2. そのデータで学習する
python -m yonmoku_nn.train_rl --data "data/rl_selfplay_gen1.jsonl" \
  --init-checkpoint checkpoints/model.pt --out checkpoints/model_rl_gen1.pt

# 3. 新しい重みを使って、また自己対戦データを集める（世代を進める）
python -m yonmoku_nn.rl_selfplay --server http://localhost:8080 \
  --checkpoint checkpoints/model_rl_gen1.pt --games 20 --simulations 100 \
  --out data/rl_selfplay_gen2.jsonl
# ... 3を繰り返す
```

`--simulations`（1手あたりのMCTSシミュレーション回数）を増やすほど自己対戦の質は上がりますが、
1手あたりの時間（シミュレーションAPI呼び出し＋ネットワーク推論の回数）も比例して増えます。
まずは少ない局数・シミュレーション回数でパイプライン全体が回ることを確認してから、
GPU環境や計算時間に応じて増やしていくのがおすすめです。

対局数の割に時間がかかる場合は `--concurrency`（ワーカープロセス数）で並列化できます。
MCTS探索自体がPython側のCPU処理（ネットワーク推論）なので、`selfplay.py`のような
スレッド並列ではなくプロセス並列にしてある（物理コア数程度まで増やすと効果が出やすい）。
YonmokuRessenサーバー側は1リクエストごとに完結するステートレスな処理なので、
複数ワーカーから同時に叩いても問題ない。

```bash
python -m yonmoku_nn.rl_selfplay --server http://localhost:8080 \
  --checkpoint checkpoints/model.pt --games 20 --simulations 100 \
  --concurrency 4 --out data/rl_selfplay_gen1.jsonl
```

### 強い方策から始める（方式A→方式Bの組み合わせ）

方式Bをランダム初期化のネットワークから始めると、序盤の自己対戦の質が低く学習が遅い。
方式A（TEST3などの模倣）で作ったチェックポイントを方式Bの`--checkpoint`/`--init-checkpoint`に
渡せば、「強いAIを模倣した状態」からMCTS強化学習を始められる。

```bash
python -m yonmoku_nn.selfplay --black-ai TEST3 --white-ai TEST3 --games 200 --out data/selfplay_test3.jsonl
python -m yonmoku_nn.train --data "data/selfplay_test3.jsonl" --out checkpoints/model.pt
# ここからcheckpoints/model.ptを使って方式Bのループ（rl_selfplay/train_rl）を回す
```

### 世代ごとの強さの確認（`yonmoku_nn.evaluate`）

方式Bを何世代か回しても、実際にTEST3などの既存AIより強くなっているかは対局してみないと
わからない（訓練データの量が少ないうちは、むしろ弱くなることもある）。`evaluate.py`は、
候補のチェックポイントをサーバーへ反映した上で既存AIと実際に対局させ、勝率を測る。

```bash
python -m yonmoku_nn.evaluate --opponent TEST3 --games 20 --candidate checkpoints/model_rl_gen5.pt
```

`--candidate`を省略すると、サーバーに現在読み込まれているモデルをそのまま評価する。
先後を交互にして対局するので、色による有利不利は打ち消される。TEST3に対する勝率が
世代を追うごとに上がっているかを見ながら、ループを続けるかどうかを判断するとよい。

NEURAL・TEST3とも着手選択が完全に決定論的（乱数を使わない）なため、`--random-opening-plies`
（既定4）で対局ごとに最初の数手をランダムに打ってから両者にAIを割り当てる。これが無いと
「NEURALが先手」「NEURALが後手」の2パターンしか実質的な対局が存在せず、`--games`を
増やしても同じ対局を繰り返すだけになってしまう。

対局の進行自体はサーバー側のスレッドで独立に進むため、`--concurrency`（既定4）で複数局を
並行して評価できる（`selfplay.py`と同じ理屈）。4局を試した実測で、直列（`--concurrency 1`）
3分17秒→並列（`--concurrency 4`）1分37秒とおよそ2倍速くなった。

## 盤面のエンコーディング（`yonmoku_nn/encoding.py`）

9×9の各マスについて、以下のチャンネル（面）を持つテンソルとして表現します（`perspective`＝その局面で着手する側の色を基準に、常に「自分/相手」で正規化）。

| index | 内容 |
|---|---|
| 0 | 自分の石 |
| 1 | 自分の石でダメージ増加マーク付き |
| 2 | 相手の石 |
| 3 | 相手の石でダメージ増加マーク付き |
| 4 | 自分の石でバックアタック補正あり |
| 5 | 相手の石でバックアタック補正あり |
| 6 | 空きマスのダメージ増加マーク |
| 7 | 空きマスの除外あとマーク |
| 8 | 空きマスの除外あとマーク（マーク付きだった＝+2相当） |
| 9 | 自分のHP/6（定数面） |
| 10 | 相手のHP/6（定数面） |
| 11 | 自分への保留ダメージ量/10（定数面） |
| 12 | 相手への保留ダメージ量/10（定数面） |

## 今後の予定（未着手）

- 強化学習ループの自動化（自己対戦→学習→新旧比較→昇格、を1コマンドで回す。今は手動で3ステップを繰り返す）。
- MCTSの並列化（現状は1手ごとに逐次シミュレーションしており、対局を複数並行させることはできるが、
  1局内のシミュレーション自体は並列化していない）。
