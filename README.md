# MiRai Server for Irodori-TTS V4

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) を、MiRaiHeadから利用できるHTTPサーバーとして起動するための互換サーバーです。

このリポジトリだけでは動作しません。先に公式のIrodori-TTSをセットアップし、音声合成できる状態にしてから、本リポジトリの2ファイルをIrodori-TTSのルートフォルダーへ配置してください。

## 主な機能

- MiRaiHead向けの `POST /tts` API
- WAV形式での音声合成結果の返却
- `voices` フォルダーからの話者自動検出
- 参照音声、潜在表現、Speaker Inversion埋め込みへの対応
- 起動時のモデル事前読み込み
- モデルリビジョンの固定による再現性の確保
- ローカルPC・同一ネットワークから接続するためのURL自動表示
- ヘルスチェック・話者一覧API

## 動作条件

- Windows 10またはWindows 11
- [uv](https://docs.astral.sh/uv/) がコマンドプロンプトから実行できること
- 公式Irodori-TTSのv4系コードと依存関係が導入済みであること
- モデルを保存できる空き容量
- Irodori-TTSを実行できるCPU/GPU環境
- 初回起動時はHugging Faceからモデルを取得できるインターネット接続（既にキャッシュ済みの場合を除く）

## セットアップ

### 1. Irodori-TTSをセットアップする

まず、[Irodori-TTS公式リポジトリ](https://github.com/Aratako/Irodori-TTS)の手順に従って環境を構築します。

例（NVIDIA CUDA 12.8環境）:

```powershell
git clone https://github.com/Aratako/Irodori-TTS.git
cd Irodori-TTS
uv sync --extra cu128
```

CPU、AMD ROCm、Intel XPUなどを使用する場合は、公式READMEから環境に合うバックエンドを選んでください。セットアップ後、公式の `infer.py` で音声生成できることを先に確認しておくことを推奨します。

### 2. サーバーファイルを配置する

このリポジトリから次の2ファイルをダウンロードします。

- `mirai_server.py`
- `start_mirai_server.bat`

2ファイルを、Irodori-TTSの `pyproject.toml`、`uv.lock`、`infer.py` と同じフォルダーへコピーします。

正しい配置例:

```text
Irodori-TTS/
├─ irodori_tts/
├─ pyproject.toml
├─ uv.lock
├─ infer.py
├─ mirai_server.py
├─ start_mirai_server.bat
└─ voices/                    # 初回起動時に自動作成
```

> フォルダーをバッチファイルへドラッグ＆ドロップする方式ではありません。2ファイル自体をIrodori-TTSのルートフォルダーへコピーしてください。

### 3. サーバーを起動する

`start_mirai_server.bat` をダブルクリックします。

正常に起動すると、ローカルPCと同一ネットワーク内のデバイスから接続できるURLが表示されます。

```text
MiRAI Head connection URL(s):
  http://127.0.0.1:5000
  http://192.168.0.15:5000
```

2行目以降のIPアドレスは環境に応じて変わります。同一ネットワーク上の別デバイスから接続するときは、`127.0.0.1` ではなく `192.168.x.x` や `10.x.x.x` など、そのPCに表示されたLANアドレスをMiRaiHeadへ設定してください。

バッチファイルは起動時にモデルを事前読み込みします。初回はモデルのダウンロードと初期化に時間がかかることがあります。次のようなログが表示されるまでウィンドウを閉じないでください。

```text
Uvicorn running on http://0.0.0.0:5000
```

終了するときは、バッチのウィンドウで `Ctrl+C` を押します。

## MiRaiHead側の接続先

MiRaiHeadとサーバーが同じPCの場合:

```text
http://127.0.0.1:5000/
```

LAN内の別PCから接続する場合:

```text
http://<サーバーPCのIPアドレス>:5000/
```

既定では `0.0.0.0` で待ち受けるため、Windowsファイアウォールの許可設定によってはLAN内からもアクセスできます。インターネットへ直接公開することは想定していません。認証機能はないため、ポート5000をルーターで外部公開しないでください。

`MIRAI_IRODORI_HOST=127.0.0.1` を指定した場合はLANから接続できないため、起動画面にもローカルPC用URLだけが表示されます。

## 話者データの配置

話者データは、Irodori-TTSルート内の `voices` フォルダーへ配置します。サーバー起動後に追加・差し替えたファイルも、次のリクエストから自動的に再検出されます。

```text
voices/
├─ character-a.speaker.safetensors
├─ character-b.pt
├─ character-c.wav
├─ character-d/
│  ├─ 01.wav
│  └─ 02.wav
└─ character-e/
   ├─ 01.pt
   └─ 02.pt
```

ファイル名の拡張子を除いた部分、またはフォルダー名がMiRaiHeadから指定する `speaker_id` になります。日本語のファイル名・フォルダー名も使用できます。

対応形式:

- Speaker Inversion: `.speaker.safetensors`
- 潜在表現: `.pt`, `.pth`
- 参照音声: `.wav`, `.flac`, `.mp3`, `.m4a`, `.ogg`, `.opus`, `.aac`, `.webm`

同じIDで複数の形式が存在する場合の優先順位:

1. `<ID>.speaker.safetensors`
2. `<ID>/` 内の複数の `.pt` / `.pth`
3. `<ID>.pt` / `<ID>.pth`
4. `<ID>/` 内の複数の参照音声
5. トップレベルの単一参照音声

`speaker_id` が空、`none`、または登録されていないIDの場合は、エラーではなく参照音声なしの推論へフォールバックします。

## API

### ヘルスチェック

```http
GET /health
```

ブラウザーで次のURLを開くことでも確認できます。

```text
http://127.0.0.1:5000/health
```

### 話者一覧

```http
GET /voices
```

```text
http://127.0.0.1:5000/voices
```

### 音声合成

```http
POST /tts
Content-Type: multipart/form-data
```

主なフォームフィールド:

| フィールド | 必須 | 既定値 | 説明 |
| --- | --- | --- | --- |
| `text` | はい | - | 読み上げる文章 |
| `speaker_id` | いいえ | 参照なし | `voices` 内の話者ID |
| `caption` | いいえ | なし | 声質・感情などの説明 |
| `num_steps` | いいえ | `12` | 推論ステップ数（1～120） |
| `cfg_scale_text` | いいえ | `3.0` | テキストCFGスケール |
| `cfg_scale_speaker` | いいえ | `5.0` | 話者CFGスケール |
| `cfg_guidance_mode` | いいえ | `independent` | `independent` / `joint` / `alternating` |
| `seed` | いいえ | ランダム | 乱数シード |
| `format` | いいえ | `wav` | 現在はWAVのみ対応 |

PowerShellからのテスト例:

```powershell
curl.exe -X POST "http://127.0.0.1:5000/tts" `
  -F "text=こんにちは。音声合成のテストです。" `
  -F "speaker_id=character-a" `
  -F "num_steps=12" `
  -F "format=wav" `
  --output output.wav
```

対話形式のAPIドキュメント:

```text
http://127.0.0.1:5000/docs
```

## 環境変数

必要に応じて、バッチファイルを起動する前に環境変数で設定を変更できます。

| 環境変数 | 既定値 | 説明 |
| --- | --- | --- |
| `MIRAI_IRODORI_HOST` | `0.0.0.0` | 待ち受けアドレス |
| `MIRAI_IRODORI_PORT` | `5000` | 待ち受けポート |
| `MIRAI_IRODORI_VOICES_DIR` | `voices` | 話者データフォルダー |
| `MIRAI_IRODORI_HF_CHECKPOINT` | `Aratako/Irodori-TTS-v4-Small` | Hugging Faceモデル |
| `MIRAI_IRODORI_CHECKPOINT` | 未設定 | ローカルのモデルファイル（指定時はHugging Faceより優先） |
| `MIRAI_IRODORI_MODEL_DEVICE` | `auto` | モデルの実行デバイス |
| `MIRAI_IRODORI_CODEC_DEVICE` | `auto` | コーデックの実行デバイス |
| `MIRAI_IRODORI_MODEL_PRECISION` | `fp32` | モデル精度 |
| `MIRAI_IRODORI_CODEC_PRECISION` | `fp32` | コーデック精度 |
| `MIRAI_IRODORI_LOG_LEVEL` | `INFO` | ログレベル |

`start_mirai_server.bat` は、検証済みのIrodori-TTS v4-Smallモデルを使うため `MIRAI_IRODORI_REVISION` を固定し、`MIRAI_IRODORI_PRELOAD=true` を設定します。

設定例:

```bat
@echo off
set "MIRAI_IRODORI_HOST=127.0.0.1"
set "MIRAI_IRODORI_PORT=5001"
call start_mirai_server.bat
```

## トラブルシューティング

### `uv could not be executed` と表示される

コマンドプロンプトを開き、次を実行してください。

```bat
uv --version
```

実行できない場合はuvをインストールし、新しいコマンドプロンプトまたはエクスプローラーを開き直してください。

### `mirai_server.py was not found` と表示される

`mirai_server.py` と `start_mirai_server.bat` を同じフォルダーへ置いてください。

### `ModuleNotFoundError` が表示される

2ファイルの配置先がIrodori-TTSのルートフォルダーであることと、公式の `uv sync --extra ...` が完了していることを確認してください。

### モデルのダウンロードに失敗する

Hugging Faceへの接続、空き容量、プロキシ設定を確認してください。途中まで取得された場合は、再度起動するとキャッシュを利用して取得を再試行します。

### ポート5000を使用できない

別のアプリがポート5000を使用している可能性があります。`MIRAI_IRODORI_PORT` に別の番号を指定してください。MiRaiHead側のURLも同じ番号へ変更します。

### CUDAのメモリ不足になる

他のGPUアプリケーションを終了してください。必要なVRAMや利用可能な量子化モデルについては、Irodori-TTS公式ドキュメントを確認してください。

## 注意事項

- 本プロジェクトはIrodori-TTS本体やモデルファイルを同梱しません。
- Irodori-TTSおよび使用するモデルのライセンス・利用条件を確認してください。
- 音声・キャラクター・話者データは、権利者の許諾と適用される規約の範囲内で使用してください。
- Irodori-TTS側の更新により互換性が変わる可能性があります。

## 関連リンク

- [Irodori-TTS](https://github.com/Aratako/Irodori-TTS)
- [Irodori-TTS v4-Small](https://huggingface.co/Aratako/Irodori-TTS-v4-Small)
- [uv](https://docs.astral.sh/uv/)
