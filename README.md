# PySCF_app

PySCF_appは、PySCF（Python for Simulating Chemistry Framework）を活用した化学計算Webアプリケーションです。
Docker環境で簡単にセットアップ・実行できます。


## 機能

- PySCFを用いた量子化学計算
- Webインターフェースによる入力・結果表示
- 計算用データの管理
- マルチスレッド並列計算（開発者向け設定として実装）

## ディレクトリ構成

```
.
├── compose.yaml         # Docker Compose 設定
├── Dockerfile           # Docker イメージ定義
├── requirements.txt     # Python依存パッケージ
├── src/
│   ├── app.py           # メインアプリケーション
│   ├── static/          # 静的ファイル（CSS, JS等）
│   └── utils/           # ユーティリティモジュール
└── data/                # 計算データ保存用
```

## セットアップ方法

1. リポジトリをクローン
   ```sh
   git clone https://github.com/kam6y/PySCF_app.git
   cd PySCF_app
   ```

2. Dockerイメージのビルドと起動
   ```sh
   docker compose up --build
   ```

3. ブラウザで `http://localhost:8501` へアクセス

### 並列計算設定

Docker環境での並列計算を調整するには:

- `compose.yaml`ファイルの`cpus`パラメータでコンテナに割り当てるCPU数を設定できます
- アプリ内の「開発者向け設定」から使用するCPUコア数を調整できます（最大値は`cpus`の設定値）
- メモリ使用量が大きくなりすぎる場合は、CPUコア数を減らすか`memory`パラメータを増やしてください

## 必要要件

- Docker, Docker Compose
- （ローカル実行の場合）Python 3.8 以上

## ライセンス

このプロジェクトはMITライセンスです。

## ToDO
・symmetry=True
・サイクルの回数を明記
・参考文献
・UVスペクトル可視化(TDDFT)
・基底関数セットと交換相関汎関数の計算の複雑さの可視化
・基底関数セットと交換相関汎関数を充実させる
・ECP（Effective Core Potential : 有効内殻ポテンシャル）(金属錯体)
・CASSCF/CASCI、NMRスペクトルの計算(拡張モジュール)
・溶媒効果

OS版で行うこと
・UVスペクトル可視化(TDDFT)
・GPU版の実装