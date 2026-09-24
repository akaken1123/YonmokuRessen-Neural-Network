# YonmokuRessen Neural Network

[YonmokuRessen](https://github.com/akaken1123/YonmokuRessen)（HP制四目並べ）用の、ニューラルネットワークによる評価・着手選択AIを学習させるためのリポジトリです。Javaの本体アプリとは別リポジトリにして、Python/PyTorchの学習環境を分離しています。

## 全体像

1. **データ収集**：YonmokuRessenのJavaサーバーを起動しておき、REST API経由でAI対AI（デフォルトAI・テストAI・テスト3AI・学習AI等）の対局を作らせる。サーバー側が既に持っている `AiVsAiDriver`（対局作成後、人手を介さず自動で最後まで打ち切る仕組み）と `/api/games/{id}/history`（着手ごとの盤面・HP・保留ダメージなどの全履歴）をそのまま使うので、盤面ルール（除外・相殺・バックアタック・ダメージ増加マーク）をPython側に再実装する必要はありません。
2. **学習**：収集した「局面 → 実際に打たれた手 → 最終的な勝敗」のデータから、PyTorchで方策（どの手を打つか）と価値（その局面がどれくらい有利か）を予測するネットワークを教師あり学習する。
3. **エクスポート**：学習済みモデルをONNX形式で書き出す。Java側での推論統合（`AiLevel`への新しいレベル追加など）は今後の作業。

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

- Java側（YonmokuRessen本体）にONNXモデルを読み込んで推論する新しい`AiLevel`を追加する。
- 教師あり学習（既存AIの模倣）だけでなく、MCTS + 自己対戦によるAlphaZero的な強化学習ループへの拡張。
