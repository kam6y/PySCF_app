import streamlit as st
import os
import multiprocessing
import platform
from utils.sidebar import molecule_sidebar
from components.main_view import show_main_view

# Docker環境のCPU数を検出する関数
def detect_docker_cpu_limit():
    """Docker環境でのCPU制限を検出する"""
    try:
        # Dockerのcgroup情報を確認
        if os.path.exists('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
            with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                quota = int(f.read().strip())
            with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                period = int(f.read().strip())
            if quota > 0:
                return max(1, quota // period)  # 利用可能CPUコア数
        
        # Docker新形式(cgroup v2)のチェック
        if os.path.exists('/sys/fs/cgroup/cpu.max'):
            with open('/sys/fs/cgroup/cpu.max', 'r') as f:
                content = f.read().strip().split()
                if content[0] != 'max':
                    quota = int(content[0])
                    period = int(content[1])
                    if quota > 0:
                        return max(1, quota // period)
    except:
        pass
    
    # 制限が検出できない場合はシステムのCPU数を返す
    return multiprocessing.cpu_count()

# --- セッションステート初期化 ---
def initialize_session_state():
    if 'xyz_string' not in st.session_state:
        st.session_state['xyz_string'] = ''
    if 'auto_reflect_to_sidebar' not in st.session_state:
        st.session_state['auto_reflect_to_sidebar'] = False
    if 'show_dft_settings' not in st.session_state:
        st.session_state['show_dft_settings'] = False
    if 'basis_set' not in st.session_state:
        st.session_state['basis_set'] = 'sto-3g'
    if 'functional' not in st.session_state:
        st.session_state['functional'] = 'b3lyp'
    if 'charge' not in st.session_state:
        st.session_state['charge'] = 0
    if 'spin' not in st.session_state:
        st.session_state['spin'] = 1
    if 'calculation_success' not in st.session_state:
        st.session_state['calculation_success'] = False
    if 'dft_result' not in st.session_state:
        st.session_state['dft_result'] = None
        
    # 溶媒効果の設定初期化
    if 'enable_solvent' not in st.session_state:
        st.session_state['enable_solvent'] = False
    if 'solvent_model' not in st.session_state:
        st.session_state['solvent_model'] = 'PCM (Polarizable Continuum Model)'
    if 'selected_solvent' not in st.session_state:
        st.session_state['selected_solvent'] = 'Water'
    if 'custom_epsilon' not in st.session_state:
        st.session_state['custom_epsilon'] = 78.3553
    
    # CPU設定
    if 'available_cpu_cores' not in st.session_state:
        # Docker環境でのCPU制限を検出
        available_cores = detect_docker_cpu_limit()
        st.session_state['available_cpu_cores'] = available_cores
        
    if 'num_cpu_cores' not in st.session_state:
        # デフォルトコア数を1に設定
        st.session_state['num_cpu_cores'] = 1
        
    # OpenMP環境変数を設定
    cpu_cores = st.session_state['num_cpu_cores']
    os.environ['OMP_NUM_THREADS'] = str(cpu_cores)

def main():
    # ページ設定
    st.set_page_config(page_title="Pyscf_Front", layout="wide")
    st.title('Pyscf_Front_DFT')
    
    # セッションの初期化
    initialize_session_state()
    
    # アプリ説明
    st.markdown("主に PySCF と Streamlit を使用")
    st.markdown("フィードバックフォームは[こちら](https://forms.gle/ZWS8CPQtHTWTPVvU7)")
    st.markdown("動画版チュートリアル[こちら](https://youtu.be/q5NOIsv2zOs)")
    st.markdown("計算時間と精度は選択した計算設定に依存します。Streamlit Cloudを使用している場合、かなり弱いCPUで計算しているので計算設定がリッチだと計算が終わらずにエラーを吐く場合があります。")
    
    # チュートリアルページの追加
    with st.expander("📚 アプリ使用ガイド（チュートリアル）", expanded=False):
        st.markdown("""
        # PySCF_Front アプリ使用ガイド
        
        このガイドでは、量子化学計算Webアプリ PySCF_Front の使用方法を解説します。最初は計算例で試してみるのがおすすめです。
        
        ## 1. 基本的な使用手順
        
        ### 1.1 分子構造の入力
        以下の3つの方法で分子構造を入力できます：
        
        #### A. PubChem名/IDからの取得
        - サイドバーで「PubChem名/IDから取得」を選択
        - 化合物名（例：水、メタノール）またはCIDを入力
        - 「xyzに変換」ボタンをクリック
        
        #### B. SMILESからの取得
        - サイドバーで「SMILESから取得」を選択
        - SMILES文字列を入力（例：HOH、CCO）
        - 「xyzに変換」ボタンをクリック
        
        #### C. XYZ座標の直接入力
        - サイドバーの「XYZ直接入力/編集」欄に直接入力
        - 第1行：原子数
        - 第2行：コメント行
        - 第3行以降：原子記号 X Y Z座標
        
        ### 1.2 計算設定
        
        #### 基底関数セット
        軽い←→重い
        - **sto-3g**: 最も軽量、初期構造の概算用
        - **3-21g, 6-31g**: 標準的な計算に適す
        - **6-31g(d), 6-31+g(d)**: より高精度の計算用
        - **6-31+g(d,p), cc-pvdz, cc-pvtz**: 高精度、計算コスト高
        
        #### 交換相関汎関数
        軽い←→重い
        - **lda**: 最も単純、金属系に適す
        - **pbe**: 幅広く使用される汎関数
        - **b3lyp, pbe0**: 標準的なハイブリッド汎関数
        - **m06, wb97x, cam-b3lyp**: 高精度、特定系に最適
        
        #### その他の設定
        - **電荷**: 分子の全電荷（例：0、-1、+1）
        - **スピン多重度**: 不対電子の状態（例：一重項=1、二重項=2）
        
        ### 1.3 溶媒効果
        
        #### 溶媒モデルの選択
        - **なし**: 気相計算
        - **IEF-PCM**: 連続誘電体モデル
        - **SMD**: より高度な溶媒和モデル
        
        #### 溶媒の選択
        - プリセット溶媒：水、DMSO、メタノール等
        - カスタム：誘電率を直接指定（IEF-PCMのみ）
        
        ### 1.4 計算の実行
        1. 分子構造と計算設定を確認
        2. 「DFT計算を開始」ボタンをクリック
        3. 進捗バーとログを確認
        4. 計算完了まで待機
        
        ## 2. 結果の見方
        
        ### 2.1 最適化後の構造
        - メイン画面に3D構造が表示
        - XYZ座標をダウンロード可能
        - 自動的にサイドバーに反映
        
        ### 2.2 分子軌道エネルギー
        - HOMO（最高被占軌道）：青色
        - LUMO（最低空軌道）：赤色
        - HOMO-LUMOギャップ：緑の矢印
        
        ### 2.3 Mulliken電荷
        - 各原子の部分電荷を表示
        - 正（陽性）・負（陰性）で色分け
        
        ### 2.4 IRスペクトル
        - 振動周波数と強度
        - 主要ピークの帰属
        - 熱力学情報も同時表示
        
        ### 2.5 分子軌道3D可視化
        - HOMO/LUMO/その他軌道を3D表示
        - 赤（正）・青（負）の等値面
        
        ### 2.6 熱力学特性
        - ゼロ点エネルギー
        - エンタルピー、エントロピー
        - 自由エネルギー
        
        ### 2.7 UV-Visスペクトル（TDDFT）
        - 吸収スペクトル計算
        - 励起エネルギーと振動子強度
        - 分子の推定色を可視化
        
        ## 3. よくある問題と対処法
        
        ### 3.1 計算時間が長い場合
        - より軽い基底関数/汎関数に変更
        - 分子サイズを確認（大きすぎる場合は分割）
        - 溶媒効果をオフに
        
        ### 3.2 エラーが発生した場合
        - 分子構造が妥当か確認
        - 電荷とスピン多重度が正しいか確認
        - 基底関数と汎関数の組み合わせを変更
        
        ### 3.3 メモリ不足
        - 開発者設定でCPUコア数を減らす
        - より小さな基底関数を使用
        
        ## 4. 高度な使用方法
        
        ### 4.1 連続計算
        1. DFT計算完了後、自動的に最適化構造が反映
        2. 計算設定を変更せずに再計算可能
        3. 異なる基底関数/汎関数で比較
        
        ### 4.2 並列計算最適化
        - Docker環境でCPUコア数を調整
        - 小分子：多コア使用で高速化
        - 大分子：少コアで安定性重視
        
        ### 4.3 結果の保存
        - XYZ構造をダウンロード
        - スペクトルデータ（CSV）をエクスポート
        - 分子軌道画像をキャプチャ
        
        ## 5. 計算例
        
        ### 簡単な例：水分子（H₂O）
        1. SMILES: "O" または PubChem: "water"
        2. 基底関数: sto-3g
        3. 汎関数: LDA
        4. 電荷: 0、スピン: 1
        
        ### 中程度の例：ベンゼン
        1. SMILES: "c1ccccc1" または PubChem: "benzene"
        2. 基底関数: 6-31g
        3. 汎関数: b3lyp
        4. 電荷: 0、スピン: 1
        
        ### 応用例：二酸化窒素
        1. PubChem: "nitrogen dioxide"
        2. 基底関数: 6-31g
        3. 汎関数: b3lyp
        4. 電荷: 0、スピン: 2(スペンが１以外だと一部の機能が使えなくなるの注意してください)
        5. TDDFT計算でUV-Visスペクトルを取得(実際に褐色の色が計算されます)
        """)
    
    # DFT計算の理論について
    with st.expander("密度汎関数理論(DFT)について", expanded=False):
        st.markdown("""
        ### 密度汎関数理論（DFT）とは
        
        密度汎関数理論（Density Functional Theory, DFT）は、多電子系の電子構造を計算するための量子力学的手法で、
        電子密度を基本変数として用います。従来のハートリー・フォック法などの波動関数ベースの方法と比較して、
        計算効率と精度のバランスに優れています。
        
        #### DFTの基本原理
        
        **ホーエンベルグ・コーンの定理**
        1. **第一定理**: 基底状態の電子密度 ρ(r) は、外部ポテンシャル V_ext(r) を一意に決定します：
           $V_{ext}(r) \\Leftrightarrow \\rho(r)$
        
        2. **第二定理**: 全エネルギー E[ρ] は正しい基底状態密度でのみ最小値をとります：
           $E[\\rho_0] \\leq E[\\rho]$
        
        **コーン・シャム方程式**
        DFTの実用的な実装では、以下のコーン・シャム方程式を解きます：
        
        $\\left[-\\frac{1}{2}\\nabla^2 + v_{eff}(r)\\right]\\psi_i(r) = \\epsilon_i\\psi_i(r)$
        
        ここで、有効ポテンシャルは：
        $v_{eff}(r) = V_{ext}(r) + \\int\\frac{\\rho(r')}{\\|r-r'\\|}dr' + v_{xc}(r)$
        
        電子密度は：
        $\\rho(r) = \\sum_{i=1}^{N}|\\psi_i(r)|^2$
        
        **エネルギー汎関数**
        全エネルギーの表現：
        $E[\\rho] = T_s[\\rho] + \\int\\rho(r)V_{ext}(r)dr + \\frac{1}{2}\\iint\\frac{\\rho(r)\\rho(r')}{\\|r-r'\\|}drdr' + E_{xc}[\\rho]$
        
        - T_s[ρ]: 非相互作用運動エネルギー
        - E_xc[ρ]: 交換相関エネルギー汎関数（近似が必要）
        
        #### DFT計算のメリット
        
        - **効率性**: N^3〜N^4のスケーリング（HFと同程度）
        - **精度**: 適切な交換相関汎関数で化学的精度を実現
        - **応用性**: 構造最適化、振動解析、励起状態計算等に対応
        - **適用範囲**: 金属、半導体、分子、表面など幅広い系に適用可能
        
        #### PySCFでのDFT実装
        
        PySCFライブラリでは、様々なDFT交換相関汎関数と基底関数を組み合わせた計算を効率的に実行できます。
        本アプリケーションでは、DFTを用いて分子の構造最適化、エネルギー計算、振動解析などを行います。
        
        #### DFTの限界と注意点
        
        - **分散相互作用**: 標準的な汎関数では van der Waals相互作用を正確に扱えない
        - **強相関系**: 多重性の高い系や強相関電子系では精度が低下
        - **自己相互作用**: 局在化状態の記述に問題が生じる場合がある
        - **バンドギャップ**: 半導体のバンドギャップを過小評価する傾向
        """, unsafe_allow_html=True)
    
    # 計算負荷の凡例
    with st.expander("計算設定の負荷について", expanded=False):
        st.markdown("""
        ### 計算負荷の目安
        各基底関数と汎関数の組み合わせによって計算時間と精度が大きく変わります。色分けは相対的な計算負荷を示しています：
        
        **基底関数の計算負荷**:
        - <span style="color:green;">**軽い**</span>: sto-3g（最小基底関数セット、計算が速いが精度は低い）
        - <span style="color:#CC9900;">**中程度**</span>: 3-21g, 6-31g（分割原子価基底関数、バランスの取れた選択肢）
        - <span style="color:orange;">**重い**</span>: 6-31g(d), 6-31+g(d)（分極関数や拡散関数を含む、より正確）
        - <span style="color:red;">**非常に重い**</span>: 6-31+g(d,p), cc-pvdz, cc-pvtz（高精度だが計算コストが高い）
        
        **汎関数の計算負荷**:
        - <span style="color:green;">**軽い**</span>: lda（局所密度近似、最も単純）
        - <span style="color:#CC9900;">**中程度**</span>: pbe（一般化勾配近似）
        - <span style="color:orange;">**重い**</span>: b3lyp, pbe0（ハイブリッド汎関数、バランスの取れた精度と効率）
        - <span style="color:red;">**非常に重い**</span>: m06, wb97x, cam-b3lyp（高度なハイブリッド汎関数、特定の系に高精度）
        
        **計算時間の目安（小分子の場合）**:
        - <span style="color:green;">緑 + 緑</span>: 数秒～数十秒
        - <span style="color:#CC9900;">黄 + 黄</span>: 数十秒～数分
        - <span style="color:orange;">オレンジ + オレンジ</span>: 数分～数十分
        - <span style="color:red;">赤 + 赤</span>: 数十分～数時間以上
        
        分子サイズが大きくなると、計算時間は原子数の3～4乗に比例して増加します。
        """, unsafe_allow_html=True)

    # 基底関数について
    with st.expander("基底関数について", expanded=False):
        st.markdown("""
        ### 基底関数とは
        
        基底関数は量子化学計算において分子軌道を表現するための数学的な基礎です。
        実際の計算では、分子軌道を基底関数の線形結合として近似します：
        
        $\\psi_i = \\sum_\\mu c_{\\mu i}\\phi_\\mu$
        
        ここで、ψ_i は分子軌道、φ_μ は基底関数、c_μi は展開係数です。
        
        #### 基底関数の数学的表現
        
        **ガウシアン型軌道（GTO）**
        
        最も一般的に使用される基底関数は、ガウシアン型軌道（Gaussian Type Orbital, GTO）です：
        
        $\\phi_{nlm}(r,\\theta,\\phi) = N \\cdot r^{n-l-1} e^{-\\alpha r^2} Y_{lm}(\\theta,\\phi)$
        
        - N: 正規化定数
        - α: 指数パラメータ（軌道の広がりを決定）
        - Y_lm: 球面調和関数
        
        **s型基底関数（l=0）**:
        $\\phi_{1s} = \\left(\\frac{2\\alpha}{\\pi}\\right)^{3/4} e^{-\\alpha r^2}$
        
        **p型基底関数（l=1）**:
        $\\phi_{2p_x} = \\left(\\frac{128\\alpha^5}{\\pi^3}\\right)^{1/4} x e^{-\\alpha r^2}$
        
        **収縮ガウシアン関数（CGF）**
        
        効率化のため、複数のガウシアン関数を線形結合して使用：
        $\\phi_{CGF} = \\sum_k d_k \\phi_{GTO,k}$
        
        #### 主な基底関数タイプの詳細
        
        **最小基底関数**:
        - **STO-3G**: スレイター型軌道を3つのガウシアンで近似
          - 内殻：1つの基底関数
          - 原子価軌道：1つの基底関数
          - 最小の関数数、高速だが精度は低い
        
        **分割原子価基底関数**:
        - **3-21G**: 内殻を3つ、原子価を2+1つのガウシアンで表現
          - 内殻：3GTO収縮関数
          - 原子価：2GTO + 1GTO
        
        - **6-31G**: より柔軟な表現
          - 内殻：6GTO収縮関数
          - 原子価：3GTO + 1GTO
        
        **分極関数を含む基底関数**:
        - **6-31G(d)**: 6-31Gに分極関数（d軌道）を追加
          $\\phi_{d} = xy \\cdot e^{-\\alpha r^2}, yz \\cdot e^{-\\alpha r^2}, \\ldots$
        
        - **6-31+G(d)**: さらに拡散関数（diffuse function）を追加
          $\\phi_{diff} = e^{-\\alpha_{small} r^2}$ （α_small << 1）
        
        **相関無矛盾基底関数（cc-pVXZ）**:
        
        Dunningの相関無矛盾基底関数は、電子相関エネルギーの収束性を考慮して設計：
        
        $E(X) = E_{CBS} + A \\cdot n^{-3}$
        
        - X = D（Double）, T（Triple）, Q（Quadruple）
        - n: ゼータ数（2, 3, 4, ...）
        
        #### 基底関数選択の指針
        
        **計算負荷のスケーリング**:
        基底関数数が N の場合、計算時間は O(N^4) にスケール
        
        **重ね合わせ行列**:
        $S_{\\mu\\nu} = \\langle\\phi_\\mu|\\phi_\\nu\\rangle$
        
        基底関数間の重なりを表し、線形独立性を確保
        
        **基底関数重ね合わせエラー（BSSE）**:
        分子間相互作用計算では補正が必要：
        $E_{BSSE} = E_{AB}^{AB} - (E_A^{AB} + E_B^{AB})$
        
        #### 基底関数選択の実践的ガイド
        
        1. **初期構造探索**: STO-3G, 3-21G
        2. **標準的な計算**: 6-31G, 6-31G(d)
        3. **高精度の構造/エネルギー**: 6-31+G(d,p), cc-pVDZ
        4. **ベンチマーク計算**: cc-pVTZ, cc-pVQZ
        5. **外推による極限値**: cc-pVDZ/TZ/QZ で外挿
        """, unsafe_allow_html=True)
    
    # 汎関数について
    with st.expander("交換相関汎関数について", expanded=False):
        st.markdown("""
        ### 交換相関汎関数とは
        
        DFT計算の中心的な近似は「交換相関汎関数」の選択です。この汎関数は電子同士の相互作用を近似的に表現し、
        計算精度に大きく影響します。様々な種類の汎関数が開発されており、系や目的に応じた選択が重要です。
        
        #### 汎関数の階層構造（Jacob's Ladder）
        
        **第1段階: 局所密度近似（LDA）**:
        
        最も単純な汎関数で、各点の電子密度のみに依存：
        $E_{xc}^{LDA}[\\rho] = \\int \\rho(r) \\epsilon_{xc}(\\rho(r)) dr$
        
        **交換エネルギー（Dirac-Slater）**:
        $E_x^{LDA}[\\rho] = -C_x \\int \\rho(r)^{4/3} dr$
        
        $C_x = \\frac{3}{4}\\left(\\frac{3}{\\pi}\\right)^{1/3}$
        
        **相関エネルギー（VWN）**:
        均一電子ガスのQMCデータをパラメータ化
        
        **第2段階: 一般化勾配近似（GGA）**:
        
        電子密度の勾配も考慮：
        $E_{xc}^{GGA}[\\rho] = \\int f(\\rho(r), \\nabla\\rho(r)) dr$
        
        **PBE汎関数**:
        $E_x^{PBE}[\\rho] = E_x^{LDA}[\\rho] \\int F_x^{PBE}(s) \\rho^{4/3} dr$
        
        $s = \\frac{|\\nabla\\rho|}{2k_F\\rho}, \\quad k_F = (3\\pi^2\\rho)^{1/3}$
        
        **第3段階: メタGGA**:
        
        運動エネルギー密度も考慮：
        $E_{xc}^{mGGA}[\\rho] = \\int f(\\rho, \\nabla\\rho, \\tau) dr$
        
        $\\tau = \\sum_i^{occ}|\\nabla\\psi_i|^2$
        
        **M06汎関数**:
        メタGGAの一種で、非共有結合相互作用を改善
        
        **第4段階: ハイブリッド汎関数**:
        
        ハートリー・フォック交換の混合：
        $E_{xc}^{hybrid} = c_x E_x^{exact} + (1-c_x)E_x^{DFT} + E_c^{DFT}$
        
        **B3LYP汎関数**:
        $E_{xc}^{B3LYP} = (1-a_0-a_x)E_x^{LDA} + a_0 E_x^{HF} + a_x E_x^{B88} + (1-a_c)E_c^{VWN} + a_c E_c^{LYP}$
        
        パラメータ: $a_0 = 0.20$, $a_x = 0.72$, $a_c = 0.81$
        
        **第5段階: 長距離補正ハイブリッド**:
        
        距離依存の交換混合：
        $E_x = (1-\\alpha)E_x^{SR,DFT}(\\mu) + \\alpha E_x^{SR,HF}(\\mu) + E_x^{LR,HF}(\\mu)$
        
        **ωB97X汎関数**:
        エラー関数による距離分離：
        $\\frac{1}{r_{12}} = \\frac{erfc(\\mu r_{12})}{r_{12}} + \\frac{erf(\\mu r_{12})}{r_{12}}$
        
        **主要汎関数の特性比較**
        
        | 汎関数 | タイプ | 特徴 | 適用例 |
        |--------|---------|------|--------|
        | LDA | 第1段階 | 最も単純、金属に適す | 金属、周期系 |
        | PBE | GGA | 幅広く使用、信頼性高い | 一般的な有機分子 |
        | B3LYP | ハイブリッド | 最も普及、バランスが良い | 有機化学全般 |
        | PBE0 | ハイブリッド | 25%のHF交換、遷移金属に適す | 無機化合物 |
        | M06 | メタGGA | 非共有結合を改善 | 弱い相互作用系 |
        | CAM-B3LYP | 長距離補正 | 電荷移動状態に適す | 励起状態、色素 |
        | ωB97X | 長距離補正 | 分散力を含む | 幅広い化学系 |
        
        #### 汎関数選択の判断基準
        
        **精度vs計算コスト**:
        - LDA < GGA < ハイブリッド < メタGGA
        - ハイブリッド汎関数は約3〜4倍のコスト
        
        **系による最適化**:
        1. **有機分子**: B3LYP, M06-2X
        2. **遷移金属**: PBE0, M06-L
        3. **励起状態**: CAM-B3LYP, LC-ωPBE
        4. **非共有結合**: M06-2X, ωB97X-D
        5. **気相/溶媒**: B3LYP/M06のペア使用
        
        #### 汎関数の制限とエラー
        
        **自己相互作用エラー（SIE）**:
        $E_{SIE} = J_{ii} - \\int\\int \\frac{\\rho_i(r)\\rho_i(r')}{|r-r'|} dr dr'$
        
        ハイブリッド汎関数で部分的に緩和
        
        **静的相関**:
        開殻多重項状態の記述には限界
        
        **デライゼーション エラー**:
        広がった系での電子密度の誤差
        
        本アプリケーションでは、系の特性に応じた最適な汎関数選択のガイドを提供します。
        """, unsafe_allow_html=True)
    
    # 溶媒効果について
    with st.expander("溶媒効果モデルについて", expanded=False):
        st.markdown("""
        ### 溶媒効果モデル
        
        実際の化学反応や分子の性質は溶媒環境に大きく影響されます。量子化学計算に溶媒効果を取り入れることで、
        より現実的な条件での分子の振る舞いを予測できます。
        
        #### 主な溶媒モデル
        
        **分極連続体モデル (PCM)**:
        - **IEF-PCM**: 媒体の表面分極を考慮した積分方程式定式化PCM。
          溶媒を誘電率で特徴づけられる連続体として扱い、溶質分子の周りに作られる分極電荷を計算。
          溶質-溶媒間の静電相互作用を効率的に計算できる。
        
        **溶媒和モデル密度 (SMD)**:
        - **SMD**: 量子力学的溶質電荷密度と原子中心のソルバトクロミック項を組み合わせたモデル。
          非電解質溶媒和、水素結合、分散相互作用などを含む、より包括的な溶媒効果を考慮できる。
        
        #### 溶媒効果計算の利点
        
        - **現実的な環境の模倣**: 実験で使用される溶媒中での分子の振る舞いをシミュレート
        - **溶媒和自由エネルギー**: 溶媒中での自由エネルギー変化を予測
        - **溶媒効果による構造変化**: 溶媒が分子構造に与える影響を評価
        - **溶液内反応性**: 溶媒が反応速度や平衡に与える影響を考慮
        
        #### 溶媒パラメータの選択
        
        - **極性溶媒**: 水（ε ≈ 78.4）、DMSO（ε ≈ 46.8）、メタノール（ε ≈ 32.6）
        - **中極性溶媒**: アセトン（ε ≈ 20.5）、ジクロロメタン（ε ≈ 8.9）
        - **非極性溶媒**: クロロホルム（ε ≈ 4.7）、トルエン（ε ≈ 2.4）
        
        溶媒効果を考慮した計算は、特に極性溶媒中での反応や、溶媒との相互作用が重要な系での計算において、
        より精度の高い結果を得るために重要です。
        """)
    
    # TDDFT（時間依存密度汎関数理論）について
    with st.expander("時間依存密度汎関数理論(TDDFT)について", expanded=False):
        st.markdown("""
        ### 時間依存密度汎関数理論（TDDFT）とは
        
        時間依存密度汎関数理論（Time-Dependent Density Functional Theory, TDDFT）は、密度汎関数理論（DFT）を拡張して
        時間に依存する系や励起状態を扱えるようにした理論です。分子の光吸収・発光特性、UV-Visスペクトル、
        電子励起状態などの計算に広く使用されています。
        
        #### TDDFTの基本概念
        
        TDDFTは、時間依存コーン・シャム方程式に基づいており、電子密度の時間発展を記述します。
        実用的な計算では、線形応答理論の枠組みで励起エネルギーと遷移確率を求めることが一般的です。
        
        主な特徴：
        
        - **励起状態の記述**: 分子の基底状態だけでなく、励起された電子状態を計算できます
        - **スペクトル計算**: UV-Vis吸収スペクトルやECDスペクトルなどの光学特性を予測できます
        - **遷移確率**: 振動子強度から吸収強度を計算し、実験的なスペクトルと比較できます
        - **状態間遷移**: 電子が遷移する軌道（HOMO→LUMOなど）を視覚化して分析できます
        
        #### TDDFTの応用
        
        - **光化学**: 光吸収によって引き起こされる化学反応の機構解明
        - **蛍光・燐光**: 発光プロセスや材料設計における発光特性の予測
        - **色素・顔料**: 分子の色（可視光吸収）の理論的予測と設計
        - **太陽電池材料**: 光エネルギー変換材料の光吸収効率の最適化
        - **光触媒**: 光誘起触媒反応の理解と新規触媒設計
        
        #### TDDFTの限界と注意点
        
        - **交換相関汎関数依存性**: 使用する汎関数に結果が大きく依存します（特にCAM-B3LYPなど長距離相互作用を
          考慮した汎関数が重要な場合があります）
        - **電荷移動状態**: 一部の交換相関汎関数では電荷移動励起状態の記述に限界があります
        - **二重励起状態**: 通常の線形応答TDDFTでは二重励起状態の記述が困難です
        - **リドベルグ状態**: 高エネルギー励起状態の記述には特殊な基底関数が必要な場合があります
        
        #### PySCFでのTDDFT実装
        
        本アプリケーションでは、PySCFのTDDFTモジュールを用いて励起状態計算を実行し、
        UV-Visスペクトルを描画するとともに、分子の予測される色を視覚化します。
        特に共役系や色素分子などの光学特性解析に有用です。
        """)
        
    st.markdown("---")
    
    # サイドバーからの入力取得
    sidebar_values = molecule_sidebar()
    
    # メインビューの表示
    show_main_view(sidebar_values)

if __name__ == "__main__":
    main()
