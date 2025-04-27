# PySCF_Front

PySCF_Frontは、PySCF（Python for Simulating Chemistry Framework）を活用した量子化学計算Webアプリケーションです。Streamlitベースのインターフェースで、密度汎関数理論（DFT）計算を簡単に実行できます。Docker環境で簡単にセットアップ・実行できます。

## 機能

- PySCFを用いた量子化学計算
  - 分子構造最適化
  - エネルギー計算
  - 分子軌道の可視化
  - IRスペクトル解析
  - 熱力学特性計算
  - 溶媒効果モデル（IEF-PCM、SMD）
- 多様な入力方法
  - SMILESから分子生成
  - PubChem名/IDから検索
  - XYZ座標の直接入力
- 計算設定
  - 多様な基底関数セット（STO-3G, 3-21G, 6-31G, 6-31G(d), 6-31+G(d), 6-31+G(d,p), cc-pVDZ, cc-pVTZ）
  - 多様な交換相関汎関数（LDA, PBE, PBE0, B3LYP, M06, ωB97X, CAM-B3LYP）
  - 計算負荷の視覚的表示
- 結果の可視化
  - 3D分子構造表示
  - 分子軌道エネルギー準位図
  - Mulliken電荷分析
  - 分子軌道の3D可視化
  - IRスペクトル（振動解析）
- マルチスレッド並列計算（開発者向け設定として実装）

## ディレクトリ構成

```
.
├── compose.yaml         # Docker Compose 設定
├── Dockerfile           # Docker イメージ定義
├── requirements.txt     # Python依存パッケージ
├── src/
│   ├── app.py           # メインアプリケーション
│   ├── components/      # UIコンポーネント
│   │   ├── main_view.py    # メイン画面
│   │   └── result_view.py  # 結果表示画面
│   └── utils/           # ユーティリティモジュール
│       ├── calculations.py  # 計算処理
│       ├── sidebar.py       # サイドバーUI
│       └── visualization.py # 可視化処理
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

## 理論的背景

アプリケーション内には、以下の理論解説が閲覧可能です：

- **密度汎関数理論(DFT)** - 多電子系の電子構造を計算するための量子力学的手法
- **基底関数** - 分子軌道を表現するための数学的基礎
  - STO-3G（軽量）から cc-pVTZ（高精度）まで様々な精度レベル
- **交換相関汎関数** - 電子相互作用を近似的に表現する関数
  - LDA（軽量）から CAM-B3LYP（高精度）まで様々な精度レベル
- **溶媒効果モデル** - 溶媒中での分子の振る舞いをシミュレート
  - IEF-PCM（分極連続体モデル）
  - SMD（溶媒和モデル密度）

## 使用方法

1. サイドバーから分子構造を入力
   - PubChem名/IDから検索
   - SMILESから生成
   - XYZ座標を直接入力

2. 計算設定を選択
   - 基底関数セット
   - 交換相関汎関数
   - 電荷
   - スピン多重度
   - （オプション）溶媒効果

3. 「DFT計算を開始」ボタンをクリック

4. 結果の分析
   - 最適化構造の確認
   - 分子軌道エネルギー準位図
   - Mulliken電荷分析
   - IRスペクトル
   - 分子軌道の3D可視化
   - 熱力学特性

## 計算負荷の目安

各基底関数と汎関数の組み合わせによる計算負荷：

- **軽い（緑）**: STO-3G + LDA（数秒～数十秒）
- **中程度（黄）**: 3-21G/6-31G + PBE（数十秒～数分）
- **重い（オレンジ）**: 6-31G(d)/6-31+G(d) + B3LYP/PBE0（数分～数十分）
- **非常に重い（赤）**: 6-31+G(d,p)/cc-pVDZ/cc-pVTZ + M06/ωB97X/CAM-B3LYP（数十分～数時間以上）

※分子サイズが大きくなると、計算時間は原子数の3～4乗に比例して増加します。

## 必要要件

- Docker, Docker Compose
- （ローカル実行の場合）Python 3.8 以上

## ToDo

- symmetry=True
- サイクルの回数を明記
- 参考文献
- UVスペクトル可視化(TDDFT)
- ECP（Effective Core Potential : 有効内殻ポテンシャル）(金属錯体)
- CASSCF/CASCI
- NMRスペクトルの計算(拡張モジュール)

OS版で行うこと
- UVスペクトル可視化(TDDFT)
- GPU版の実装

## ライセンス

このプロジェクトはMITライセンスです。