import streamlit as st
import pandas as pd
import plotly.express as px
import py3Dmol
import tempfile
import os
from pyscf import tools

def show_analysis_tabs(result, atoms):
    tabs = st.tabs(["軌道・電荷・IR", "分子軌道可視化"])
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
            ir_result = result['run_and_plot_ir_spectrum'](result['mf_opt'], result['mol'], atoms)
            if ir_result['detail']:
                st.error(f'IRスペクトル計算中にエラーが発生しました: {ir_result["detail"]}')
            else:
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
    with tabs[1]:
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
            
            # 一時CUBEファイル生成
            with tempfile.TemporaryDirectory() as tmpdir:
                cube_path = os.path.join(tmpdir, f"{orbital_name}.cube")
                tools.cubegen.orbital(mol, cube_path, mf.mo_coeff[:, orbital_index], nx=80, ny=80, nz=80)
                with open(cube_path, 'r') as f:
                    cube_data = f.read()
            
            # xyz生成
            BOHR_TO_ANGSTROM = 0.529177210903
            atom_syms = [a[0] for a in mol._atom]
            atom_coords = mol.atom_coords() * BOHR_TO_ANGSTROM
            xyz_block = f"{len(atom_syms)}\n\n" + '\n'.join([
                f"{atom_syms[i]} {atom_coords[i][0]:.6f} {atom_coords[i][1]:.6f} {atom_coords[i][2]:.6f}" for i in range(len(atom_syms))
            ])
            
            # py3Dmolで可視化
            view = py3Dmol.view(width=max, height=400)
            view.addModel(xyz_block, 'xyz')
            view.setStyle({'stick': {}})
            view.addVolumetricData(cube_data, 'cube', {'isoval': 0.02, 'color': 'red', 'opacity': 0.75})
            view.addVolumetricData(cube_data, 'cube', {'isoval': -0.02, 'color': 'blue', 'opacity': 0.75})
            view.zoomTo()
            st.components.v1.html(view._make_html(), height=420)
            
        except Exception as e:
            st.error(f"分子軌道可視化中にエラーが発生しました: {str(e)}")
            st.warning("XYZ座標を編集した場合は、再度「DFT計算を開始」ボタンを押して新しい構造で計算を実行してください。")
