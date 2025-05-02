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
    tabs = st.tabs(["軌道・電荷・IR", "分子軌道可視化", "熱力学特性", "UV-Visスペクトル", "参考文献"])
    
    # --- カスタムツールチップ用CSSを挿入 ---
    st.markdown("""
    <style>
    .tooltip {
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted #555;
      cursor: help;
      font-size: 1.6em;
    }
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 50vw;
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
      font-size: 0.6em;
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
        
    with tabs[3]:
        show_tddft_tab(result)
        
    with tabs[4]:
        show_references_tab()

def show_orbital_charges_ir_tab(result, atoms):
    """軌道・電荷・IRタブの表示"""
    st.markdown(
        '''<span class="tooltip">分子軌道エネルギー準位図
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
        '''<span class="tooltip">Mulliken電荷
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
        '''<span class="tooltip">IR Spectrum (Vibrational Analysis)
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
            error_msg = ir_result["detail"]
            st.error(f'IRスペクトル計算中にエラーが発生しました: {error_msg}')
            
            # スピン多重度に関連するエラーメッセージを追加
            if "too many values to unpack" in error_msg:
                spin_value = result['mol'].spin + 1 if hasattr(result['mol'], 'spin') else "不明"
                st.warning(f"このエラーはスピン多重度（現在: {spin_value}）の設定が原因で発生した可能性があります。スピン多重度を1に設定して再計算することをお勧めします。")
            elif "スピン多重度" in error_msg:
                st.warning("高いスピン多重度を使用する場合は、サイドバーで新しい計算を開始する際にスピン多重度を1に設定してください。")
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
        '''<span class="tooltip">分子軌道3D可視化
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
            st.caption(f"{orbital_name} 軌道番号: {orbital_index} エネルギー: {orbital_energies[orbital_name]:.3f} eV")
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
        '''<span class="tooltip">熱力学特性
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
            # スピン多重度のチェック
            from pyscf import dft
            if hasattr(mol, 'spin') and mol.spin > 0 and isinstance(mf, dft.rks.RKS):
                # RKSでスピン多重度が1より大きい場合は警告
                st.error(f"スピン多重度が{mol.spin+1}の場合、熱力学特性計算にはUKS計算が必要です。")
                st.warning("スピン多重度を1に設定して再計算するか、適切なUKS計算を使用してください。")
                return
                
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
            
            try:
                # ヘシアン計算器を初期化
                hess = rks.Hessian(mf)
                
                # ヘシアン行列を計算
                hess_matrix = hess.kernel()
                
                # 振動解析
                freq_info = thermo.harmonic_analysis(mol, hess_matrix)
                
                # 熱力学特性の計算 (298.15 K, 1 atm)
                thermo_info = thermo.thermo(mf, freq_info['freq_au'], 298.15, 101325)
            except ValueError as e:
                if "too many values to unpack" in str(e):
                    st.error(f"熱力学特性の計算に失敗しました: {str(e)}")
                    st.warning("現在、熱力学特性の計算はスピン多重度を1のときのみ可能です。")
                    return
                else:
                    raise e
            
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
    

def show_references_tab() -> None:
    """参考文献タブの表示"""
    st.markdown(
        '''
        <span class="tooltip">参考文献
          <span class="tooltiptext">
            このアプリケーションの開発と理論的背景に関連する主要文献です。
            各セクションはテーマごとに「査読論文 → 書籍 → オンラインリソース」の順で並びます。
          </span>
        </span>
        ''',
        unsafe_allow_html=True
    )

    # ───────────────────────────────────────
    # 1. PySCF フレームワーク
    # ───────────────────────────────────────
    with st.expander("PySCF フレームワーク", expanded=True):
        st.markdown("""
### 査読論文
1. **Q. Sun *et&nbsp;al*.**, “Recent developments in the PySCF program package”, *J. Chem. Phys.* **153**, 024109 (2020).  
2. **Q. Sun *et&nbsp;al*.**, “PySCF: The Python-based Simulations of Chemistry Framework”, *WIREs Comput. Mol. Sci.* **8**, e1340 (2018).  

### オンラインリソース
- PySCF 公式サイト (最終アクセス 2025-04-28): <https://pyscf.org/>
        """)

    # ───────────────────────────────────────
    # 2. Density-Functional Theory (DFT)
    # ───────────────────────────────────────
    with st.expander("密度汎関数理論（DFT）"):
        st.markdown("""
### 査読論文
1. **U. Ekström *et&nbsp;al*.**, “Arbitrary-order density-functional response theory from automatic differentiation”, *J. Chem. Theory Comput.* **6**, 1971 (2010).
        """)

    # ───────────────────────────────────────
    # 3. 交換相関汎関数
    # ───────────────────────────────────────
    with st.expander("交換相関汎関数"):
        st.markdown("""
### 査読論文
1. **J. P. Perdew *et&nbsp;al*.**, “Generalized Gradient Approximation Made Simple”, *Phys. Rev. Lett.* **77**, 3865 (1996).  
2. **A. D. Becke**, “Density-functional thermochemistry III. The role of exact exchange”, *J. Chem. Phys.* **98**, 5648 (1993).  
3. **Y. Zhao, D. G. Truhlar**, “The M06 suite of density functionals…”, *Theor. Chem. Acc.* **120**, 215 (2008).  
4. **S. Lehtola *et&nbsp;al*.**, “Recent developments in libxc — A comprehensive library of functionals for DFT”, *SoftwareX* **7**, 1 (2018).
        """)

    # ───────────────────────────────────────
    # 4. 基底関数セット
    # ───────────────────────────────────────
    with st.expander("基底関数セット"):
        st.markdown("""
### 査読論文
1. **W. J. Hehre *et&nbsp;al*.**, “Self-Consistent MO Methods IX. An Extended Gaussian-Type Basis…”, *J. Chem. Phys.* **54**, 724 (1971).  
2. **T. H. Dunning Jr.**, “Gaussian basis sets for use in correlated molecular calculations I…”, *J. Chem. Phys.* **90**, 1007 (1989).
        """)

    # ───────────────────────────────────────
    # 5. 溶媒効果モデル
    # ───────────────────────────────────────
    with st.expander("溶媒効果モデル"):
        st.markdown("""
### 査読論文
1. **B. Mennucci**, “Polarizable continuum model”, *WIREs Comput. Mol. Sci.* **2**, 386 (2012).  
2. **A. V. Marenich *et&nbsp;al*.**, “Universal solvation model based on solute electron density… (SMD)”, *J. Phys. Chem. B* **113**, 6378 (2009).  
3. **F. Lipparini *et&nbsp;al*.**, “Quantum, classical, and hybrid QM/MM calculations in solution: ddCOSMO…”, *J. Chem. Phys.* **141**, 184108 (2014).
        """)

    # ───────────────────────────────────────
    # 6. 分子構造最適化
    # ───────────────────────────────────────
    with st.expander("分子構造最適化"):
        st.markdown("""
### 査読論文
1. **L.-P. Wang, C.-C. Song**, “Geometry optimization made simple with translation and rotation coordinates”, *J. Chem. Phys.* **144**, 214108 (2016).
        """)

    # ───────────────────────────────────────
    # 7. 振動解析（IR）
    # ───────────────────────────────────────
    with st.expander("振動解析（IR）"):
        st.markdown("""
### 査読論文
1. **P. Pulay**, “Ab initio calculation of molecular vibrational intensities”, *J. Mol. Struct.* **80**, 17 (1980).  
2. **V. Barone**, “Vibrational zero-point energies and thermodynamic functions beyond the harmonic approximation”, *J. Chem. Phys.* **120**, 3059 (2004).  
3. **J. A. Barnes *et&nbsp;al*.**, “Solvent effect on vibrational frequencies…”, *J. Mol. Spectrosc.* **192**, 86 (1998).  
4. **R. Improta *et&nbsp;al*.**, “Geometries and properties of excited states… TD-DFT/PCM”, *J. Chem. Phys.* **124**, 124504 (2006).
        """)

    # ───────────────────────────────────────
    # 8. 可視化ツール
    # ───────────────────────────────────────
    with st.expander("可視化ツール"):
        st.markdown("""
### 査読論文
1. **N. Rego, D. Koes**, “3Dmol.js: molecular visualization with WebGL”, *Bioinformatics* **31**, 1322 (2015).
        """)

    # ───────────────────────────────────────
    # 9. GPU 版 PySCF
    # ───────────────────────────────────────
    with st.expander("GPU 版 PySCF"):
        st.markdown("""
### 査読論文
1. **J. Wu *et&nbsp;al*.**, “Enhancing GPU-acceleration in the Python-based Simulations of Chemistry Framework”, *arXiv preprint* arXiv:2404.09452 (2024).
        """)

    # ───────────────────────────────────────
    # 10. 参考書（和書）
    # ───────────────────────────────────────
    with st.expander("参考書（和書）"):
        st.markdown("""
- **西長亨・本田康**, 『有機化学者のための量子化学計算入門 ― Gaussian の基本と有効利用のヒント ―』, 第2版, 化学同人 (2014).
        """)

    # ───────────────────────────────────────
    # 11. オンラインリソース
    # ───────────────────────────────────────
    with st.expander("オンラインリソース"):
        st.markdown("""
- PySCF GPU4PySCF README (最終アクセス 2024-04-28): <https://github.com/pyscf/gpu4pyscf>  
- py3Dmol in Jupyter チュートリアル (最終アクセス 2024-04-28): <https://birdlet.github.io/2019/10/02/py3dmol_example/>
        """)


def show_tddft_tab(result):
    """UV-Visスペクトルタブの表示"""
    import pandas as pd
    import plotly.graph_objects as go
    import numpy as np
    import streamlit as st
    from scipy.constants import physical_constants
    from utils.calculations import apply_solvent_effects

    # result_idを取得（タブのユニーク性を確保するため）
    result_id = st.session_state.get('result_view_id', id(result))
    
    # run_tddft_calculation関数を直接定義
    def run_tddft_calculation(mf_opt, mol, n_states=10, solvent_settings=None):
        """
        TDDFT計算を実行し、UV-Visスペクトルを計算する
        
        Args:
            mf_opt: 最適化後のメイン電子状態計算オブジェクト
            mol: 分子オブジェクト
            n_states: 計算する励起状態の数
            solvent_settings: 溶媒効果の設定
            
        Returns:
            dictionary: TDDFT計算結果を含む辞書
        """
        from pyscf import tddft, lib
        import os
        
        # 並列計算設定
        cpu_cores = st.session_state.get('num_cpu_cores', 1)
        os.environ['OMP_NUM_THREADS'] = str(cpu_cores)
        lib.num_threads(cpu_cores)
        
        # 計算リソース設定
        mem_per_core = 2000  # コアあたりのメモリ使用量（MB）
        total_memory = mem_per_core * cpu_cores
        
        # メモリ設定
        mol.max_memory = total_memory
        mf_opt.max_memory = total_memory
        
        # 溶媒効果の適用（設定されている場合）
        if solvent_settings and solvent_settings.get('enable_solvent', False):
            try:
                mf_opt, _ = apply_solvent_effects(mf_opt, solvent_settings)
            except Exception as e:
                st.warning(f"TDDFT計算に溶媒効果を適用できませんでした: {str(e)}")
        
        # TDDFT計算
        mytd = tddft.TDDFT(mf_opt)
        mytd.nstates = n_states
        mytd.max_memory = total_memory
        
        try:
            # TDDFT計算実行
            mytd.kernel()
            
            # 振動子強度の取得（NaN は 0 に置換）
            osc_strengths = mytd.oscillator_strength()[:n_states]
            osc_strengths = np.nan_to_num(osc_strengths)
            
            # 励起エネルギーをハートリー単位から eV へ変換
            ha_2_ev = physical_constants['Hartree energy in eV'][0]
            energies_ev = mytd.e[:n_states] * ha_2_ev
            
            # 波長（nm）と波数（cm-1）に変換
            energies_nm = 1239.841984 / energies_ev
            energies_cm = 8065.54429 * energies_ev
            
            # 電子状態遷移情報の抽出
            nocc = np.count_nonzero(mf_opt.mo_occ > 0)
            nmo = mf_opt.mo_coeff.shape[1]
            nvirt = nmo - nocc  # 仮想軌道数
            
            transitions = []  # 各要素は (occupied_orbital, virtual_orbital, coefficient)
            for state_idx in range(n_states):
                try:
                    xvec_raw = mytd.xy[state_idx][0]
                    xvec = np.array(xvec_raw)
                    xvec_flat = xvec.flatten()
                    max_index = np.argmax(np.abs(xvec_flat))
                    max_coeff = xvec_flat[max_index]
                    occ_index = max_index // nvirt
                    virt_index = max_index % nvirt
                    transitions.append((occ_index, nocc + virt_index, max_coeff))
                except Exception as e:
                    transitions.append((None, None, None))
            
            # スペクトル表示範囲を設定（可視光領域380-780nmを含みつつ、計算された全ピークも含める）
            if len(energies_nm) > 0:
                min_nm = min(380, np.min(energies_nm) - 100)  # 可視光領域開始か計算値-100の小さい方
                max_nm = max(780, np.max(energies_nm) + 100)  # 可視光領域終了か計算値+100の大きい方
            else:
                min_nm = 380
                max_nm = 780
                
            # 連続スペクトル作成用のx軸範囲設定
            min_ev = 1239.841984 / max_nm  # 波長から電子ボルトに変換
            max_ev = 1239.841984 / min_nm
            x_range = np.linspace(min_ev, max_ev, num=1000)
            
            # コーシー分布の定義
            def cauchy(x, x0, gamma):
                """コーシー分布（正規化済み）"""
                return (1 / np.pi) * (gamma / ((x - x0)**2 + gamma**2))
            
            # 連続スペクトルの作成（コーシー分布の重ね合わせ）
            spectral_width = 0.1  # コーシー分布の幅
            intensity = np.zeros(x_range.size)
            for e, f in zip(energies_ev, osc_strengths):
                intensity += cauchy(x_range, e, spectral_width) * f
            
            # スペクトルの正規化（最大値を1に）
            if np.max(intensity) > 0:
                intensity = intensity / np.max(intensity)
            
            # 結果の辞書を生成
            result = {
                'success': True,
                'n_states': n_states,
                'energies_ev': energies_ev,
                'energies_nm': energies_nm,
                'energies_cm': energies_cm,
                'osc_strengths': osc_strengths,
                'transitions': transitions,
                'x_range_ev': x_range,
                'x_range_nm': 1239.841984 / x_range,
                'x_range_cm': 8065.54429 * x_range,
                'intensity': intensity,
                'functional': mf_opt.xc if hasattr(mf_opt, 'xc') else 'Unknown',
                'min_nm': min_nm,
                'max_nm': max_nm
            }
            
            return result
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    st.markdown(
        '''<span class="tooltip">UV-Visスペクトル計算
        <span class="tooltiptext">
          時間依存密度汎関数理論（TDDFT）を用いて分子の励起状態と紫外可視スペクトルを計算します。
          この計算では分子の電子励起エネルギーと遷移双極子モーメントから吸収スペクトルを予測します。
          計算結果は実験的なUV-Visスペクトルと比較できます。
        </span>
      </span>''',
        unsafe_allow_html=True
    )
    
    # セッションステートの初期化
    if 'tddft_result' not in st.session_state:
        st.session_state['tddft_result'] = None
    if 'tddft_running' not in st.session_state:
        st.session_state['tddft_running'] = False
    if 'tddft_n_states' not in st.session_state:
        st.session_state['tddft_n_states'] = 10
    
    # 計算設定
    n_states = st.number_input(
        '計算する励起状態数', 
        min_value=1, 
        max_value=30, 
        value=st.session_state['tddft_n_states'],
        help='より多くの状態を計算するとより広いスペクトル範囲をカバーできますが、計算負荷が増加します',
        key=f'tddft_states_input_{result_id:x}'
    )
    
    # 値の変更を検出してセッションに保存（計算は開始しない）
    if n_states != st.session_state['tddft_n_states']:
        st.session_state['tddft_n_states'] = n_states

    st.markdown(f"**使用する基底関数・汎関数はDFT計算時のものを使います。**")
    
    # 溶媒設定の表示
    solvent_info = "なし"
    if result.get('solvent_info'):
        solvent_info = result.get('solvent_info')
    st.markdown(f"**溶媒効果**: {solvent_info}")
    
    # 計算実行ボタン
    calc_button = st.button('TDDFT計算を実行', key=f'run_tddft_button_{result_id:x}')
    
    if calc_button:
        st.session_state['tddft_running'] = True
        with st.spinner('TDDFT計算実行中... この計算は数分〜数日かかることがあります'):
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
                # TDDFT計算実行
                tddft_result = run_tddft_calculation(
                    result['mf_opt'],
                    result['mol'],
                    n_states=st.session_state['tddft_n_states'],
                    solvent_settings=solvent_settings
                )
                st.session_state['tddft_result'] = tddft_result
                st.session_state['tddft_running'] = False
                st.rerun()  # 計算完了後に表示を更新
            except Exception as e:
                st.error(f"TDDFT計算中にエラーが発生しました: {str(e)}")
                st.session_state['tddft_running'] = False
    
    # 計算結果の表示
    tddft_result = st.session_state.get('tddft_result')
    
    if st.session_state.get('tddft_running'):
        # 計算中の表示
        st.info('TDDFT計算実行中... しばらくお待ちください。')
    elif tddft_result is not None:
        if tddft_result.get('success', False):
            
            # スペクトルデータをDataFrameに変換（連続スペクトル）
            spectral_data = {
                "Excitation Energy (eV)": tddft_result['x_range_ev'].tolist(),
                "Intensity": tddft_result['intensity'].tolist(),
                "Exchange-Correlation Functional": [tddft_result['functional']] * len(tddft_result['x_range_ev']),
                "Excitation Energy (nm)": tddft_result['x_range_nm'].tolist(),
                "Excitation Energy (cm-1)": tddft_result['x_range_cm'].tolist()
            }
            df_spectral = pd.DataFrame(spectral_data)
            
            # 各励起状態のデータをDataFrameに変換
            osc_data = {
                "Exchange-Correlation Functional": [],
                "State": [],
                "Excitation Energy (eV)": [],
                "Excitation Energy (nm)": [],
                "Excitation Energy (cm-1)": [],
                "Oscillator Strength": [],
                "Occupied Orbital": [],
                "Virtual Orbital": [],
                "Coefficient": []
            }
            
            for i, (energy_ev, energy_nm, energy_cm, osc, trans) in enumerate(
                zip(tddft_result['energies_ev'], tddft_result['energies_nm'], 
                    tddft_result['energies_cm'], tddft_result['osc_strengths'], 
                    tddft_result['transitions'])):
                
                occ, virt, coeff = trans
                osc_data["Exchange-Correlation Functional"].append(tddft_result['functional'])
                osc_data["State"].append(i+1)
                osc_data["Excitation Energy (eV)"].append(energy_ev)
                osc_data["Excitation Energy (nm)"].append(energy_nm)
                osc_data["Excitation Energy (cm-1)"].append(energy_cm)
                osc_data["Oscillator Strength"].append(osc)
                osc_data["Occupied Orbital"].append(occ)
                osc_data["Virtual Orbital"].append(virt)
                osc_data["Coefficient"].append(coeff)
            
            df_osc = pd.DataFrame(osc_data)
            
            # Plotlyでインタラクティブなスペクトル表示
            fig = go.Figure()
            
            # eV単位のトレース
            fig.add_trace(go.Scatter(
                x=df_spectral["Excitation Energy (eV)"],
                y=df_spectral["Intensity"],
                mode='lines',
                name='eV',
                line=dict(color='blue', width=2)
            ))
            
            # 各励起状態を縦線で表示（eV）
            for i, row in df_osc.iterrows():
                if row["Oscillator Strength"] > 0.01:  # 振動子強度が一定以上の状態のみ表示
                    fig.add_trace(go.Scatter(
                        x=[row["Excitation Energy (eV)"], row["Excitation Energy (eV)"]],
                        y=[0, row["Oscillator Strength"] / df_osc["Oscillator Strength"].max()],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        name=f'State {int(row["State"])}',
                        showlegend=False
                    ))
            
            # nm単位のトレース
            fig.add_trace(go.Scatter(
                x=df_spectral["Excitation Energy (nm)"],
                y=df_spectral["Intensity"],
                mode='lines',
                name='nm',
                line=dict(color='green', width=2),
                visible=False
            ))
            
            # 各励起状態を縦線で表示（nm）
            for i, row in df_osc.iterrows():
                if row["Oscillator Strength"] > 0.01:
                    fig.add_trace(go.Scatter(
                        x=[row["Excitation Energy (nm)"], row["Excitation Energy (nm)"]],
                        y=[0, row["Oscillator Strength"] / df_osc["Oscillator Strength"].max()],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        showlegend=False,
                        visible=False
                    ))
            
            # cm^-1単位のトレース
            fig.add_trace(go.Scatter(
                x=df_spectral["Excitation Energy (cm-1)"],
                y=df_spectral["Intensity"],
                mode='lines',
                name='cm⁻¹',
                line=dict(color='purple', width=2),
                visible=False
            ))
            
            # 各励起状態を縦線で表示（cm^-1）
            for i, row in df_osc.iterrows():
                if row["Oscillator Strength"] > 0.01:
                    fig.add_trace(go.Scatter(
                        x=[row["Excitation Energy (cm-1)"], row["Excitation Energy (cm-1)"]],
                        y=[0, row["Oscillator Strength"] / df_osc["Oscillator Strength"].max()],
                        mode='lines',
                        line=dict(color='red', width=1, dash='dash'),
                        showlegend=False,
                        visible=False
                    ))
            
            # ボタンとドロップダウンメニューの作成
            n_peaks = sum(1 for f in df_osc["Oscillator Strength"] if f > 0.01)
            
            # 表示範囲用（380-750nm固定）
            display_min_nm = 380
            display_max_nm = 750
            
            # 単位切り替えボタンを追加
            updatemenus = [
                dict(
                    active=0,
                    buttons=list([
                        dict(label="eV",
                            method="update",
                            args=[{"visible": [True] + [True] * n_peaks + 
                                          [False] * (n_peaks) + [False] * (n_peaks + 1)},
                                  {"title": "UV-Vis Spectrum (eV)",
                                   "xaxis": {"title": "Excitation Energy (eV)"}}]),
                        dict(label="nm",
                            method="update",
                            args=[{"visible": [False] + [False] * n_peaks + 
                                          [True] * (n_peaks + 1) + [False] * n_peaks},
                                  {"title": "UV-Vis Spectrum (nm)",
                                   "xaxis": {"title": "Wavelength (nm)", 
                                            "autorange": "reversed", 
                                            "range": [display_max_nm, display_min_nm]}}]),
                        dict(label="cm⁻¹",
                            method="update",
                            args=[{"visible": [False] * (n_peaks + 1) + 
                                          [False] * (n_peaks + 1) + [True] * (n_peaks + 1)},
                                  {"title": "UV-Vis Spectrum (cm⁻¹)",
                                   "xaxis": {"title": "Wavenumber (cm⁻¹)"}}]),
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
            
            # レイアウト設定
            fig.update_layout(
                title="UV-Vis Spectrum (eV)",
                xaxis_title="Excitation Energy (eV)",
                yaxis_title="Normalized Intensity",
                updatemenus=updatemenus,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                ),
                margin=dict(l=50, r=50, t=80, b=50),
            )
            
            # スペクトルの表示
            st.plotly_chart(fig, use_container_width=True)
            
            # 各励起状態の詳細情報を表として表示
            st.markdown("**励起状態の詳細**")
            
            # 振動子強度でソートして表示
            df_osc_sorted = df_osc.sort_values(by="Oscillator Strength", ascending=False).reset_index(drop=True)
            df_display = df_osc_sorted[["State", "Excitation Energy (eV)", "Excitation Energy (nm)", 
                                     "Oscillator Strength", "Occupied Orbital", "Virtual Orbital"]]
            
            # 表示用にカラム名を日本語化
            df_display.columns = ["状態", "励起エネルギー (eV)", "波長 (nm)", 
                                "振動子強度", "占有軌道", "仮想軌道"]
            
            # 表の表示
            st.dataframe(df_display, use_container_width=True)
            
            # データのダウンロードボタン
            csv_spectral = df_spectral.to_csv(index=False).encode('utf-8')
            csv_osc = df_osc.to_csv(index=False).encode('utf-8')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="連続スペクトルデータをダウンロード",
                    data=csv_spectral,
                    file_name="spectral_data.csv",
                    mime="text/csv",
                    key=f'download_spectral_{result_id:x}'
                )
            with col2:
                st.download_button(
                    label="励起状態データをダウンロード",
                    data=csv_osc,
                    file_name="oscillator_strengths.csv",
                    mime="text/csv",
                    key=f'download_osc_{result_id:x}'
                )
            
            # === 分子の色可視化セクション ===
            from utils.calculations import calculate_molecule_color
            
            st.markdown("---")
            st.markdown(
            '''<span class="tooltip">分子の色の可視化
            <span class="tooltiptext">
            このセクションでは、計算された吸収スペクトルから分子の色を推定します。
            吸光度の強度を調整して色の濃さを変更できます
            </span>
            </span>''',
                unsafe_allow_html=True
            )
            
            # 初期値の設定（より適切な初期値を使用）
            if 'color_multiplier' not in st.session_state:
                st.session_state['color_multiplier'] = 5.0  # デフォルト値を調整
            
            # 色計算の実行
            color_result = calculate_molecule_color(tddft_result, multiplier=st.session_state['color_multiplier'])
            
            if color_result.get('success', False):
                # 2列レイアウト
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # カラーパッチの表示
                    st.image(f"data:image/png;base64,{color_result['color_img']}", caption="推定色")
                    
                    # 吸光度強度調整用数値入力ボックス（col1内に配置）
                    multiplier = st.number_input(
                        '吸光度強度倍率:',
                        min_value=0.0,
                        max_value=100.0,
                        value=st.session_state['color_multiplier'],
                        step=1.0,
                        format="%.1f",
                        key=f'color_multiplier_input_{result_id:x}',
                        help='吸光度の強度を調整します。大きい値ほど色が濃くなります。小さい値では薄く、大きい値では濃くなります。'
                    )
                    
                    # セッションに保存し、値が変わったらリロード
                    if st.session_state['color_multiplier'] != multiplier:
                        st.session_state['color_multiplier'] = multiplier
                        st.rerun()  # UIを更新して色を再計算する
                    
                    # RGB値とHEX値の表示
                    rgb = color_result['rgb']
                    hex_color = color_result['rgb_hex']
                    st.markdown(f"""
                    **RGB値**: [{int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}]  
                    **HEX値**: {hex_color}
                    """)
                
                with col2:
                    # スペクトルと等色関数のグラフ
                    fig = go.Figure()
                    
                    # 吸収スペクトル（計算結果から）
                    fig.add_trace(go.Scatter(
                        x=color_result['wavelength_range'],
                        y=color_result['absorption'],
                        mode='lines',
                        name='吸収スペクトル',
                        line=dict(color='purple', width=2, dash='dash')
                    ))
                    
                    # 透過率スペクトル
                    fig.add_trace(go.Scatter(
                        x=color_result['wavelength_range'],
                        y=color_result['transmittance'],
                        mode='lines',
                        name='透過率(倍率適用後)',
                        line=dict(color='orange', width=2)
                    ))
                    
                    # X, Y, Z等色関数（透過率と掛け合わせたもの）
                    wavelength = color_result['wavelength_range']
                    transmittance = np.array(color_result['transmittance'])
                    
                    # 各等色関数と透過率の掛け算
                    xBar = np.array(color_result['xBar']) * transmittance
                    yBar = np.array(color_result['yBar']) * transmittance
                    zBar = np.array(color_result['zBar']) * transmittance
                    
                    fig.add_trace(go.Scatter(
                        x=wavelength, y=xBar,
                        mode='lines', name='X×透過率',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=wavelength, y=yBar,
                        mode='lines', name='Y×透過率',
                        line=dict(color='green', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=wavelength, y=zBar,
                        mode='lines', name='Z×透過率',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f'分光透過率と等色関数 (倍率: {st.session_state["color_multiplier"]:.1f})',
                        xaxis_title='波長 (nm)',
                        yaxis_title='相対強度',
                        legend=dict(
                            x=0.98,        # 右端寄せ
                            y=0.98,        # 上端寄せ
                            xanchor='right',  # x 座標を右端基準に
                            yanchor='top',    # y 座標を上端基準に
                        ),
                        margin=dict(l=0, r=0, t=40, b=0),
                        xaxis=dict(range=[380, 780]),
                        yaxis=dict(range=[0, 1.05])
                    )

                    st.plotly_chart(fig, use_container_width=True)
                
                # 色の意味に関する注意事項
                st.info("""
                **注意**: 表示されている色は計算された吸収スペクトルから推定されたものであり、実際の物質の色とは異なる場合があります。
                溶媒や濃度、観察条件によって色の見え方は大きく変わることがあります。
                """)
                
            else:
                # 色計算失敗時のエラーメッセージ
                st.error(f"色計算に失敗しました: {color_result.get('error', '不明なエラー')}")
            
        else:
            # 計算失敗時のエラーメッセージ表示
            st.error(f"TDDFT計算に失敗しました: {tddft_result.get('error', '不明なエラー')}")
    else:
        # 計算前の状態
        st.info('「TDDFT計算を実行」ボタンをクリックすると、UV-Visスペクトル計算を実行します。')
        st.warning('注意: TDDFT計算は計算負荷が高く、分子サイズや計算設定によっては数分～数十分かかる場合があります。')


# テスト実行用
if __name__ == "__main__":
    st.set_page_config(page_title="Reference Tab Demo")
    show_references_tab()


