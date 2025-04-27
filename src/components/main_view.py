import streamlit as st
from utils.visualization import visualize_molecule_3d
from utils.calculations import run_dft_calculation
from utils.sidebar import get_molecule_from_xyz

def show_main_view(sidebar_values):
    """メインビューの表示"""
    # 設定値の取り出し
    input_method = sidebar_values['input_method']
    pubchem_query = sidebar_values['pubchem_query']
    smiles = sidebar_values['smiles']
    basis_set = sidebar_values['basis_set']
    functional = sidebar_values['functional']
    charge = sidebar_values['charge']
    spin = sidebar_values['spin']
    xyz_string = sidebar_values['xyz_string']
    
    # XYZの統合
    xyz_final = st.session_state['xyz_string']
    
    atoms, coords = None, None
    if xyz_final:
        atoms, coords = get_molecule_from_xyz(xyz_final)
    
    # メインコンテンツの表示
    if st.session_state['xyz_string']:
        atoms, coords = get_molecule_from_xyz(st.session_state['xyz_string'])
        if atoms is not None and coords is not None:
            calc_container = st.container()
            
            if calc_container.button('DFT計算を開始', key='run_dft'):
                # 現在の計算設定をセッションに保存
                st.session_state['basis_set'] = basis_set
                st.session_state['functional'] = functional
                st.session_state['charge'] = charge
                st.session_state['spin'] = spin
                
                # 溶媒効果の設定を保存
                solvent_settings = sidebar_values.get('solvent_settings', {})
                if solvent_settings:
                    st.session_state['enable_solvent'] = solvent_settings.get('enable_solvent', False)
                    st.session_state['solvent_model'] = solvent_settings.get('solvent_model', '')
                    st.session_state['selected_solvent'] = solvent_settings.get('selected_solvent', '')
                    st.session_state['custom_epsilon'] = solvent_settings.get('epsilon', 0)
                
                # 設定表示フラグをオン（計算開始前に設定）
                st.session_state['show_dft_settings'] = True
                
                # 溶媒情報の取得（計算画面にも表示する）
                solvent_info_display = ""
                if st.session_state.get('enable_solvent', False):
                    solvent_model = st.session_state.get('solvent_model', '')
                    selected_solvent = st.session_state.get('selected_solvent', 'カスタム')
                    
                    if solvent_model == 'IEF-PCM':
                        if selected_solvent == 'カスタム':
                            epsilon = st.session_state.get('custom_epsilon', 0.0)
                            if epsilon is not None:
                                solvent_info_display = f"- 溶媒効果: **{solvent_model}** (カスタム, ε={epsilon:.4f})"
                            else:
                                solvent_info_display = f"- 溶媒効果: **{solvent_model}** (カスタム)"
                        else:
                            solvent_info_display = f"- 溶媒効果: **{solvent_model}** ({selected_solvent})"
                    elif solvent_model == 'SMD':
                        if selected_solvent == 'カスタム':
                            solvent_info_display = f"- 溶媒効果: **{solvent_model}** (Water)"
                        else:
                            solvent_info_display = f"- 溶媒効果: **{solvent_model}** ({selected_solvent})"
                
                # 計算設定をボタン押下時に表示（計算実行前）
                calc_container.info(f"""
                **計算設定:**
                - 基底関数セット: **{basis_set}**
                - 交換相関汎関数: **{functional}**
                - 電荷: {charge}
                - スピン多重度: {spin}
                {solvent_info_display}
                """)
                
                # 溶媒設定を確認（内部処理用）
                solvent_settings = sidebar_values.get('solvent_settings', None)
                
                # 計算中の状態をメイン画面に表示（簡略化）
                with st.spinner('DFT計算実行中...'):
                    try:
                        # CPUコア数の取得
                        cpu_cores = sidebar_values.get('cpu_cores', st.session_state.get('num_cpu_cores', 1))
                        
                        # 計算実行
                        result = run_dft_calculation(
                            atoms, 
                            coords, 
                            basis_set, 
                            functional, 
                            charge, 
                            spin-1,
                            cpu_cores=cpu_cores,
                            solvent_settings=sidebar_values.get('solvent_settings', None)
                        )
                        
                        st.session_state['dft_result'] = result
                        
                        # 計算結果をセッションに保存し、サイドバーに自動反映
                        st.session_state['xyz_string'] = result['xyz_optimized']
                        
                        # サイドバーへの自動反映のフラグを設定（sidebarのxyz入力欄を更新するため）
                        st.session_state['auto_reflect_to_sidebar'] = True
                        
                        # 成功メッセージを表示（直接表示）
                        calc_container.success(f'構造最適化計算が完了しました。XYZ座標をサイドバーに自動反映します。')
                        
                        # ページを再読み込み（streamlit 0.88.0以降の新しい方法）
                        st.rerun()
                        
                    except Exception as e:
                        error_message = str(e)
                        if "TypeError: first argument must be a string" in error_message or "gto.basis.load_basis" in error_message:
                            st.error(f"基底関数セットのエラーが発生しました: {error_message}\n\n選択した基底関数 '{basis_set}' がPySCFで正しく処理できなかった可能性があります。\n\n一般的な解決策:\n1. 標準的な表記を使用しているか確認してください\n2. より一般的な基底関数（'sto-3g'や'6-31g'など）を試してみてください")
                        elif "KeyError" in error_message and "xc_code" in error_message:
                            st.error(f"汎関数のエラーが発生しました: {error_message}\n\n選択した汎関数 '{functional}' がPySCFで正しく処理できなかった可能性があります。\n\n一般的な解決策:\n1. 標準的な表記を使用しているか確認してください\n2. より一般的な汎関数（'b3lyp'や'pbe0'など）を試してみてください")
                        else:
                            st.error(f'計算中にエラーが発生しました: {error_message}')
            
            # 成功メッセージの表示（計算完了後のページリロード時）
            if st.session_state.get('calculation_success', False):
                calc_container.success('計算が完了しました。XYZ座標をサイドバーに自動反映します。')
                # メッセージを表示した後はフラグをリセット
                st.session_state['calculation_success'] = False
            
            # 3D分子構造の見出し
            st.subheader('3D分子構造')
            
            # 3D分子構造の表示
            view = visualize_molecule_3d(atoms, coords)
            st.components.v1.html(view._make_html(), height=370)
            
            # 結果表示部分
            show_results(atoms, input_method, pubchem_query, smiles)
        else:
            st.info('有効な分子構造を入力してください。')
    else:
        st.info('有効な分子構造をサイドバーから入力してください。')

def show_results(atoms, input_method, pubchem_query, smiles):
    """計算結果の表示"""
    from components.result_view import show_result_view
    
    # --- DFT計算結果をセッションに保存し、ダウンロードボタンは常に表示 ---
    if 'dft_result' not in st.session_state:
        st.session_state['dft_result'] = None
        
    result = st.session_state['dft_result']
    if result is not None:
        st.markdown("---")
        st.subheader('最適化後の分子構造 (XYZ形式)')
        st.markdown('**計算結果は自動的にサイドバーに反映されています。このXYZデータで続けて計算ができます。**')
        st.text(result['xyz_optimized'])
        
        # --- xyz_filenameを先に定義 ---
        if input_method == 'SMILESから取得' and smiles:
            xyz_filename = f"{smiles}.xyz"
        elif input_method == 'PubChem名/IDから取得' and pubchem_query:
            xyz_filename = f"{pubchem_query}.xyz"
        else:
            xyz_filename = "molecule_optimized.xyz"
            
        # --- ダウンロードボタンのみに変更 ---
        st.download_button(
            label=f"最適化後のXYZファイルをダウンロード ({xyz_filename})",
            data=result['xyz_optimized'],
            file_name=xyz_filename,
            mime='text/plain'
        )
        
        st.markdown("---")
        
        # 計算結果が現在の分子構造と一致するか確認
        result_atoms_count = len(result['mol']._atom)
        current_atoms_count = len(atoms) if atoms is not None else 0
        
        if result_atoms_count != current_atoms_count:
            st.warning(f"""
            **注意**: 現在の分子構造が計算結果と一致しません。
            - 計算時の原子数: {result_atoms_count}
            - 現在の原子数: {current_atoms_count}
            
            分析結果を表示するには、再度「DFT計算を開始」ボタンを押して新しい構造で計算を実行してください。
            """)
        else:
            # 分析タブを表示
            show_result_view(result, atoms)
