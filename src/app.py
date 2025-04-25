import streamlit as st
from utils.sidebar import molecule_sidebar
from components.main_view import show_main_view

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

def main():
    # ページ設定
    st.set_page_config(page_title="分子構造DFT計算", layout="wide")
    st.title('分子構造DFT計算')
    
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
