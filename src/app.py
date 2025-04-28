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
    st.markdown("計算時間と精度は選択した計算設定に依存します。Streamlit Cloudを使用している場合、かなり弱いCPUで計算しているので計算設定がリッチだと計算が終わらずにエラーを吐く場合があります。")
    

    # DFT計算の理論について
    with st.expander("密度汎関数理論(DFT)について", expanded=False):
        st.markdown("""
        ### 密度汎関数理論（DFT）とは
        
        密度汎関数理論（Density Functional Theory, DFT）は、多電子系の電子構造を計算するための量子力学的手法で、
        電子密度を基本変数として用います。従来のハートリー・フォック法などの波動関数ベースの方法と比較して、
        計算効率と精度のバランスに優れています。
        
        #### DFTの基本原理
        
        DFTは以下の2つのホーエンベルグ・コーンの定理に基づいています：
        
        1. **第一定理**: 基底状態の電子密度は、外部ポテンシャル（原子核による）を一意に決定する
        2. **第二定理**: 任意の電子密度に対するエネルギー汎関数が存在し、正しい基底状態の電子密度でのみ最小値をとる
        
        コーン・シャム方程式では、相互作用する多電子系を、同じ電子密度を持つ非相互作用系に置き換えて計算します。
        実際の計算では、交換相関汎関数と呼ばれるエネルギー項を近似する必要があります。
        
        #### DFT計算のメリット
        
        - **効率性**: 電子の波動関数ではなく電子密度を扱うため、計算コストが比較的小さい
        - **精度**: 適切な交換相関汎関数を選ぶことで、高い精度が得られる
        - **応用性**: 分子構造の最適化、振動解析、電子状態の計算など、幅広い化学的性質の予測が可能
        
        #### PySCFでのDFT実装
        
        PySCFライブラリでは、様々なDFT交換相関汎関数と基底関数を組み合わせた計算を効率的に実行できます。
        本アプリケーションでは、DFTを用いて分子の構造最適化、エネルギー計算、振動解析などを行います。
        """)
    
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
        実際の計算では、分子軌道を基底関数の線形結合として近似します。
        基底関数セットの選択は計算精度と計算コストに大きく影響します。
        
        #### 主な基底関数タイプ
        
        **最小基底関数**:
        - **STO-3G**: 最小限の基底関数セット。各原子価軌道に1つの関数のみを使用。
          計算は速いですが精度は低め。主に初期構造の探索や大きな分子の概算に使用。
        
        **分割原子価基底関数**:
        - **3-21G**: 内殻軌道は3つのガウス関数で、原子価軌道は2つと1つのガウス関数で表現。
          STO-3Gよりも優れた精度を提供し、計算コストも比較的低い。
        - **6-31G**: 内殻軌道は6つのガウス関数で、原子価軌道は3つと1つのガウス関数で表現。
          分子構造や基本的な性質の計算に広く使用される標準的な選択肢。
        
        **分極関数を含む基底関数**:
        - **6-31G(d)**: 6-31Gに重原子のd軌道を追加。(d)や*と表記されることもある。
          結合や反応における電子の分極効果をより正確に表現。
        - **6-31+G(d)**: さらに拡散関数(+)を追加。陰イオンやラジカルなどの拡張電子系に重要。
        - **6-31+G(d,p)**: 水素原子にもp軌道の分極関数を追加。より正確な構造や振動数の計算に有用。
        
        **相関無矛盾基底関数**:
        - **cc-pVDZ**: 二重ゼータ品質の相関無矛盾基底関数。
          電子相関を含む高精度計算に適している。
        - **cc-pVTZ**: 三重ゼータ品質で、さらに高精度だが計算コストも高い。
        
        #### 基底関数選択の目安
        
        - **初期構造探索**: STO-3G, 3-21G
        - **標準的な計算**: 6-31G
        - **高精度の構造/エネルギー**: 6-31G(d), 6-31+G(d)
        - **厳密な計算/特殊な系**: 6-31+G(d,p), cc-pVDZ, cc-pVTZ
        """)
    
    # 汎関数について
    with st.expander("交換相関汎関数について", expanded=False):
        st.markdown("""
        ### 交換相関汎関数とは
        
        DFT計算の中心的な近似は「交換相関汎関数」の選択です。この汎関数は電子同士の相互作用を近似的に表現し、
        計算精度に大きく影響します。様々な種類の汎関数が開発されており、系や目的に応じた選択が重要です。
        
        #### 汎関数の種類
        
        **局所密度近似 (LDA)**:
        - **LDA**: 一様電子ガスモデルに基づく最も単純な汎関数。
          各点での交換相関エネルギーは、その点の電子密度のみに依存。
          金属や周期系に対して比較的良い結果を与えるが、分子系では結合エネルギーを過大評価する傾向あり。
        
        **一般化勾配近似 (GGA)**:
        - **PBE**: Perdew-Burke-Ernzerhofによる汎関数。
          電子密度の勾配を考慮し、LDAよりも精度が向上。特に固体物理学での使用に適している。
        
        **ハイブリッド汎関数**:
        - **B3LYP**: 最も広く使用されているハイブリッド汎関数。
          Beckeの3パラメータ汎関数とLee-Yang-Parrの相関汎関数を組み合わせたもの。
          ハートリー・フォック交換を約20%含む。多くの化学系で良好な結果を示す。
        - **PBE0**: PBE汎関数とハートリー・フォック交換を混合（25%）。
          特に無機化合物や遷移金属化合物に対して良い精度を示す。
        
        **メタGGA汎関数**:
        - **M06**: Miaoによる高度な汎関数ファミリー。運動エネルギー密度も考慮する。
          非共有結合相互作用を含む系での計算に適している。
        
        **長距離補正ハイブリッド汎関数**:
        - **CAM-B3LYP**: クーロン減衰法を用いたB3LYPの拡張版。
          電荷移動状態や長距離相互作用が重要な系での使用に適している。
        - **ωB97X**: 長距離補正と分散力補正を含む高精度汎関数。
          広範囲の化学反応や非共有結合に対して良好な結果を与える。
        
        #### 汎関数選択の目安
        
        - **標準的な有機分子**: B3LYP
        - **遷移金属やイオンを含む系**: PBE0
        - **弱い相互作用が重要な系**: M06, ωB97X
        - **励起状態や電荷移動**: CAM-B3LYP
        
        適切な汎関数の選択は研究対象や目的に大きく依存するため、必要に応じて複数の汎関数での計算結果を比較検討することも重要です。
        """)
    
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
    
    st.markdown("---")
    
    # サイドバーからの入力取得
    sidebar_values = molecule_sidebar()
    
    # メインビューの表示
    show_main_view(sidebar_values)

if __name__ == "__main__":
    main()
