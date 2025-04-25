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
    st.title('Pyscf_Front')
    
    # セッションの初期化
    initialize_session_state()
    
    # アプリ説明
    st.markdown("主に PySCF と Streamlit を使用")
    st.markdown("計算時間と精度は選択した計算設定に依存します。かなり弱いCPUで計算しているので計算設定がリッチだと計算が終わらずにエラーを吐く場合があります。")
    st.markdown("---")
    
    # サイドバーからの入力取得
    sidebar_values = molecule_sidebar()
    
    # メインビューの表示
    show_main_view(sidebar_values)

if __name__ == "__main__":
    main()
