import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from utils.visualization import plot_mo_energies_plotly, visualize_molecular_orbital
from utils.calculations import run_and_plot_ir_spectrum

def show_result_view(result, atoms):
    """計算結果の分析タブ表示"""
    # 必要な関数をresultに追加して渡す
    result['plot_mo_energies'] = plot_mo_energies_plotly
    result['run_and_plot_ir_spectrum'] = run_and_plot_ir_spectrum
    
    # エラーハンドリングを追加
    try:
        show_analysis_tabs(result, atoms)
    except Exception as e:
        st.error(f'分析タブの表示中にエラーが発生しました：{str(e)}')
        st.warning('XYZ座標を編集した場合は、再度「DFT計算を開始」ボタンを押して新しい構造で計算を実行してください。')

def show_analysis_tabs(result, atoms):
    """分析タブの表示"""
    tabs = st.tabs(["軌道・電荷・IR", "分子軌道可視化", "熱力学特性"])
    
    # --- カスタムツールチップ用CSSを挿入 ---
    st.markdown("""
    <style>
    .tooltip {
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted #555;
      cursor: help;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 30vw;
      background-color: #222;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 8px 12px;
      position: absolute;
      z-index: 1;
      top: 100%;
      left: 0;
      opacity: 0;
      transition: opacity 0.3s;
      font-size: 0.7em;
    }
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with tabs[0]:
        show_orbital_charges_ir_tab(result, atoms)
    
    with tabs[1]:
        show_molecular_orbital_tab(result)
        
    with tabs[2]:
        show_thermodynamics_tab(result, atoms)

def show_orbital_charges_ir_tab(result, atoms):
    """軌道・電荷・IRタブの表示"""
    st.markdown(
        '''<span class="tooltip" style="font-size:1.2em;">分子軌道エネルギー準位図
  <span class="tooltiptext">
    分子軌道エネルギー準位図は、各分子軌道のエネルギーを可視化したものです。青色の線はHOMO（最高被占有分子軌道）、赤色の線はLUMO（最低空分子軌道）を示し、緑色の矢印は電子励起に必要な最小エネルギー差（HOMO-LUMOギャップ）を表します。
  </span>
</span>''',
        unsafe_allow_html=True
    )
    
    homo_idx = result['mol'].nelectron // 2 - 1
    fig = result['plot_mo_energies'](result['orbital_energies'], homo_idx, gap=result['gap'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    
    st.markdown(
        '''<span class="tooltip" style="font-size:1.2em;">Mulliken電荷
  <span class="tooltiptext">
    Mulliken電荷は、各原子における電子の偏り（部分電荷）を示します。正の値は電子が少なく陽性、負の値は電子が多く陰性であることを意味します。棒グラフで各原子の電荷分布を視覚的に比較できます。
  </span>
</span>''',
        unsafe_allow_html=True
    )
    
    # 原子リストと電荷データの長さを確認し、不一致の場合はエラーメッセージを表示
    try:
        atoms_len = len(atoms) if atoms is not None else 0
        charges_len = len(result['charges']) if 'charges' in result and result['charges'] is not None else 0
        
        if atoms_len != charges_len:
            st.warning(f"原子データと電荷データの長さが一致しません（原子: {atoms_len}, 電荷: {charges_len}）。再計算が必要です。")
        else:
            charges_df = pd.DataFrame({
                '原子': atoms,
                '電荷': result['charges']
            })
            fig_charge = px.bar(charges_df, x='原子', y='電荷', color='電荷', color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_charge, use_container_width=True)
            st.dataframe(charges_df)
    except Exception as e:
        st.error(f"Mulliken電荷の表示中にエラーが発生しました: {str(e)}")
        st.warning("XYZ座標を編集した場合は、再度「DFT計算を開始」ボタンを押して新しい構造で計算を実行してください。")
    
    st.markdown("---")
    
    st.markdown(
        '''<span class="tooltip" style="font-size:1.2em;">IR Spectrum (Vibrational Analysis)
  <span class="tooltiptext">
    IRスペクトル（赤外振動スペクトル）は、分子の振動モードごとの赤外吸収強度と周波数を示します。主なピークは分子内の特定の結合や構造に対応し、分子の構造解析や同定に利用されます。構造最適化が問題なく出来ているかの確認のために行います。エラーを吐いた場合は条件を見直してください。
  </span>
</span>''',
        unsafe_allow_html=True
    )
    
    try:
        # 溶媒設定をセッションから取得
        solvent_settings = None
        if st.session_state.get('enable_solvent', False):
            solvent_settings = {
                'enable_solvent': st.session_state.get('enable_solvent', False),
                'solvent_model': st.session_state.get('solvent_model', ''),
                'selected_solvent': st.session_state.get('selected_solvent', ''),
                'epsilon': st.session_state.get('custom_epsilon', 0)
            }
            
        ir_result = result['run_and_plot_ir_spectrum'](
            result['mf_opt'], 
            result['mol'], 
            atoms, 
            solvent_settings=solvent_settings
        )
        if ir_result['detail']:
            st.error(f'IRスペクトル計算中にエラーが発生しました: {ir_result["detail"]}')
        else:
            # セッションにIR計算結果を保存（熱力学特性計算で使用）
            st.session_state['ir_result'] = ir_result
            
            st.text(ir_result['thermo_text'])
            st.image(f"data:image/png;base64,{ir_result['img_base64']}")
            st.markdown('主なIRピークと振動モードの対応')
            if ir_result['mode_df'] is not None:
                st.dataframe(ir_result['mode_df'], use_container_width=True, height=350)
            else:
                st.warning("IRスペクトルの詳細データ取得に失敗しました。")
    except Exception as e:
        st.error(f"IRスペクトル分析中にエラーが発生しました: {str(e)}")
        st.warning("XYZ座標を編集した場合は、再度「DFT計算を開始」ボタンを押して新しい構造で計算を実行してください。")

def show_molecular_orbital_tab(result):
    """分子軌道可視化タブの表示"""
    st.markdown(
        '''<span class="tooltip" style="font-size:1.2em;">分子軌道3D可視化
  <span class="tooltiptext">
    分子軌道（HOMO, LUMO, その前後）を3D等値面で可視化します。<br>
    ドロップダウンから軌道を選択すると、該当軌道の電子雲分布（CUBEファイル）が3D表示されます。<br>
    赤色は正の等値面、青色は負の等値面を表します。分子構造は最適化後のものを使用します。
  </span>
</span>''',
        unsafe_allow_html=True
    )
    
    try:
        mf = result['mf_opt']
        mol = result['mol']
        homo_idx = mol.nelectron // 2 - 1
        lumo_idx = homo_idx + 1
        
        # 可視化する軌道リスト
        orbitals = {
            "LUMO+3": lumo_idx + 3,
            "LUMO+2": lumo_idx + 2,
            "LUMO+1": lumo_idx + 1,
            "LUMO": lumo_idx,
            "HOMO": homo_idx,
            "HOMO-1": homo_idx - 1,
            "HOMO-2": homo_idx - 2,
            "HOMO-3": homo_idx - 3
        }
        
        hartree_to_ev = 27.2114
        orbital_energies = {name: mf.mo_energy[idx] * hartree_to_ev for name, idx in orbitals.items() if 0 <= idx < len(mf.mo_energy)}
        
        # 選択肢をフィルタ
        available_orbitals = [(f"{name} ({orbital_energies[name]:.3f} eV)", name) for name in orbitals if name in orbital_energies]
        
        if not available_orbitals:
            st.warning("表示可能な分子軌道がありません。計算結果が無効になっている可能性があります。再計算をお試しください。")
            return
            
        selected = st.selectbox('可視化する軌道', available_orbitals, format_func=lambda x: x[0])
        orbital_name = selected[1]
        orbital_index = orbitals[orbital_name]
        
        # 分子軌道の可視化結果をキャッシュするためのキー
        # molオブジェクトのid（メモリアドレス）とorbital_indexを組み合わせてユニークなキーを生成
        cache_key = f"orbital_viz_{id(mol)}_{orbital_index}"
        
        # すでに計算されたデータがあればキャッシュから取得
        if cache_key in st.session_state:
            viz_result = st.session_state[cache_key]
        else:
            # 分子軌道の可視化を実行
            viz_result = visualize_molecular_orbital(mol, mf, orbital_index, orbital_name)
            # 結果をキャッシュに保存
            st.session_state[cache_key] = viz_result
        
        if viz_result['success']:
            view = viz_result['view']
            st.components.v1.html(view._make_html(), height=520)
            st.caption(f"{orbital_name} エネルギー: {orbital_energies[orbital_name]:.3f} eV")
        else:
            st.error(f"分子軌道の可視化に失敗しました: {viz_result['error']}")
            
    except Exception as e:
        st.error(f"分子軌道可視化中にエラーが発生しました: {str(e)}")
        st.warning("XYZ座標を編集した場合は、再度「DFT計算を開始」ボタンを押して新しい構造で計算を実行してください。")

def show_thermodynamics_tab(result, atoms=None):
    """熱力学特性タブの表示"""
    import pandas as pd
    import traceback
    import os
    from pyscf.hessian import thermo, rks
    from pyscf import lib
    
    st.markdown(
        '''<span class="tooltip" style="font-size:1.2em;">熱力学特性
  <span class="tooltiptext">
    熱力学特性は、分子の安定性とエネルギーに関する重要な情報を提供します。ゼロ点エネルギー、エンタルピー、エントロピー、自由エネルギーなどが含まれます。これらの値は温度298.15 K、圧力1気圧での計算結果です。
  </span>
</span>''',
        unsafe_allow_html=True
    )
    
    mol = result['mol']
    mf = result['mf_opt']
    
    # 熱力学計算結果をセッションに保存してキャッシュする
    # molオブジェクトのid（メモリアドレス）を使用してユニークなキーを生成
    thermo_key = f"thermo_result_{id(mol)}"
    
    # 既に計算済みの場合はキャッシュから取得
    if thermo_key in st.session_state:
        try:
            cached_thermo = st.session_state[thermo_key]
            display_thermo_results(cached_thermo['thermo_info'], cached_thermo['freq_info'])
            return
        except Exception as e:
            st.warning(f"キャッシュデータの読み込みエラー: {e}。再計算します。")
    
    # まだ計算していない場合は計算を実行
    with st.spinner('熱力学特性を計算中...この処理には時間がかかることがあります'):
        try:
            # 並列計算設定
            cpu_cores = st.session_state.get('num_cpu_cores', 1)
            
            # OpenMP環境変数を明示的に設定
            os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
            
            # PySCF並列設定
            lib.num_threads(cpu_cores)
            
            # ヘシアン計算を実行して直接振動数を取得
            # 並列計算に合わせてメモリ設定
            mem_per_core = 2000  # コアあたりのメモリ使用量（MB）
            total_memory = mem_per_core * cpu_cores
            
            mol.max_memory = total_memory
            mf.max_memory = total_memory
            
            # ヘシアン計算器を初期化
            hess = rks.Hessian(mf)
            
            # ヘシアン行列を計算
            hess_matrix = hess.kernel()
            
            # 振動解析
            freq_info = thermo.harmonic_analysis(mol, hess_matrix)
            
            # 熱力学特性の計算 (298.15 K, 1 atm)
            thermo_info = thermo.thermo(mf, freq_info['freq_au'], 298.15, 101325)
            
            # 計算結果をセッションに保存
            st.session_state[thermo_key] = {
                'thermo_info': thermo_info,
                'freq_info': freq_info
            }
            
            # 結果を表示
            display_thermo_results(thermo_info, freq_info)
            
        except Exception as e:
            st.error(f"熱力学特性の計算中にエラーが発生しました: {str(e)}")
            st.code(traceback.format_exc())
            st.warning("分子が大きすぎるか複雑すぎる場合は、計算に時間がかかるか、メモリ不足で失敗する可能性があります。より少ないCPUコア数を試すと改善する場合があります。")

def display_thermo_results(thermo_info, freq_info):
    """熱力学特性の計算結果を表示する"""
    import pandas as pd
    
    # 単位の変換定数
    HARTREE_TO_EV = 27.211386
    HARTREE_TO_KCAL = 627.5095
    
    # 熱力学特性データの作成
    thermo_data = {
        "物理量": [
            "温度 (K)",
            "圧力 (atm)",
            "電子エネルギー (Hartree)",
            "ゼロ点エネルギー補正 (Hartree)",
            "熱エネルギー補正 (Hartree)",
            "熱エンタルピー補正 (Hartree)",
            "熱自由エネルギー補正 (Hartree)",
            "電子エネルギー + ゼロ点エネルギー (Hartree)",
            "電子エネルギー + 熱エネルギー補正 (Hartree)",
            "電子エネルギー + 熱エンタルピー補正 (Hartree)",
            "電子エネルギー + 熱自由エネルギー補正 (Hartree)",
            "熱容量 (Cv) (J/mol·K)",
            "エントロピー (S) (J/mol·K)",
            "電子エネルギー + ゼロ点エネルギー (eV)",
            "電子エネルギー + ゼロ点エネルギー (kcal/mol)"
        ],
        "値": [
            thermo_info['temperature'][0],
            thermo_info['pressure'][0] / 101325,  # Paからatmへ変換
            thermo_info['E0'][0],
            thermo_info['ZPE'][0],
            thermo_info['E_vib'][0],
            thermo_info['H_vib'][0],
            thermo_info['G_vib'][0],
            thermo_info['E_0K'][0],
            thermo_info['E_tot'][0],
            thermo_info['H_tot'][0],
            thermo_info['G_tot'][0],
            thermo_info['Cv_tot'][0],
            thermo_info['S_tot'][0],
            thermo_info['E_0K'][0] * HARTREE_TO_EV,
            thermo_info['E_0K'][0] * HARTREE_TO_KCAL
        ]
    }
    
    # DataFrameの作成
    df_thermo = pd.DataFrame(thermo_data)
    
    # 表示
        # 主要な熱力学特性を強調表示
    st.markdown("### 主要な熱力学特性")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="ゼロ点エネルギー (ZPE)", 
            value=f"{thermo_info['ZPE'][0]:.6f} Hartree"
        )
        st.metric(
            label="熱エンタルピー (H)", 
            value=f"{thermo_info['H_tot'][0]:.6f} Hartree"
        )
    
    with col2:
        st.metric(
            label="熱自由エネルギー (G)", 
            value=f"{thermo_info['G_tot'][0]:.6f} Hartree"
        )
        st.metric(
            label="エントロピー (S)", 
            value=f"{thermo_info['S_tot'][0]:.6f} J/mol·K"
        )

    st.dataframe(df_thermo, use_container_width=True, height=600)
    


