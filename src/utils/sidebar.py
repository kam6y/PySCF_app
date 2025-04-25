import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_molecule_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    atoms = []
    coords = []
    for atom in mol.GetAtoms():
        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
        atoms.append(atom.GetSymbol())
        coords.append([pos.x, pos.y, pos.z])
    return mol, atoms, np.array(coords)

def get_molecule_from_xyz(xyz_string):
    lines = xyz_string.strip().split('\n')
    try:
        n_atoms = int(lines[0])
        atoms = []
        coords = []
        for i in range(2, 2 + n_atoms):
            if i >= len(lines):
                break
            parts = lines[i].split()
            if len(parts) >= 4:
                atom_symbol = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append(atom_symbol)
                coords.append([x, y, z])
        return atoms, np.array(coords)
    except:
        return None, None

def molecule_sidebar():
    # 計算設定情報をサイドバーの一番上に表示（計算実行後のみ）
    if st.session_state.get('show_dft_settings', False):
        st.sidebar.info(f"""
        **現在のDFT計算設定:**
        - 基底関数セット: **{st.session_state['basis_set']}**
        - 交換相関汎関数: **{st.session_state['functional']}**
        - 電荷: {st.session_state['charge']}
        - スピン多重度: {st.session_state['spin']}
        """)
    
    st.sidebar.title('分子構造入力')
    input_method = st.sidebar.radio('**入力方法**', ['PubChem名/IDから取得', 'SMILESから取得'], index=0)
    pubchem_query = ''
    smiles = ''
    xyz_string = st.session_state.get('xyz_string', '')
    if input_method == 'PubChem名/IDから取得':
        pubchem_query = st.sidebar.text_input('化合物名またはCID', key='pubchem_sidebar')
    else:
        smiles = st.sidebar.text_input('SMILES', key='smiles_sidebar')
    # 変換ボタンを一番下に
    convert_btn = st.sidebar.button('xyzに変換', key='convert_btn')
    # 変換・反映ロジック
    if convert_btn:
        if input_method == 'PubChem名/IDから取得' and pubchem_query:
            import pubchempy as pcp
            try:
                c = pcp.get_compounds(pubchem_query, 'name')
                if not c:
                    c = pcp.get_compounds(pubchem_query, 'cid')
                if c:
                    mol = c[0]
                    smiles_from_pubchem = mol.isomeric_smiles or mol.canonical_smiles
                    if smiles_from_pubchem:
                        mol_rdkit, atoms, coords = get_molecule_from_smiles(smiles_from_pubchem)
                        if mol_rdkit is not None:
                            xyz = f"{len(atoms)}\nPubChemから取得(RDKit再構造化)\n" + '\n'.join([
                                f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" for a, c in zip(atoms, coords)
                            ])
                            st.session_state['xyz_string'] = xyz
                            xyz_string = xyz  # XYZ欄にも反映
                            st.sidebar.success('PubChemから取得・3D再構造化成功')
                        else:
                            st.sidebar.error('SMILESから3D構造生成に失敗しました')
                    else:
                        st.sidebar.error('PubChemからSMILESが取得できませんでした')
                else:
                    st.sidebar.error('該当する化合物が見つかりません')
            except Exception as e:
                st.sidebar.error(f'PubChem取得エラー: {e}')
        elif input_method == 'SMILESから取得' and smiles:
            mol, atoms, coords = get_molecule_from_smiles(smiles)
            if mol is not None:
                xyz = f"{len(atoms)}\nSMILESから生成\n" + '\n'.join([
                    f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" for a, c in zip(atoms, coords)
                ])
                st.session_state['xyz_string'] = xyz
                xyz_string = xyz  # XYZ欄にも反映
                st.sidebar.success('SMILESからXYZ変換成功')
            else:
                st.sidebar.error('SMILESの解釈に失敗しました')
        else:
            st.session_state['xyz_string'] = xyz_string
    # XYZ直接入力/編集欄
    st.sidebar.markdown('**XYZ直接入力/編集**')
    
    # 計算後に自動反映されたXYZ値を取得
    xyz_string = st.session_state.get('xyz_string', '')
    
    # XYZ直接入力/編集欄をセッションステートの現在の値で初期化
    xyz_input_key = 'xyz_input_sidebar'
    new_xyz_string = st.sidebar.text_area('XYZ座標', xyz_string, height=200, key=xyz_input_key)
    
    # XYZ座標が変更された場合、古い計算結果をクリア
    if new_xyz_string != st.session_state.get('xyz_string', ''):
        if 'dft_result' in st.session_state:
            st.session_state['dft_result'] = None
        st.session_state['xyz_string'] = new_xyz_string
    
    # 自動反映フラグがある場合、サイドバーに反映して処理完了を示すようにフラグをリセット
    if st.session_state.get('auto_reflect_to_sidebar', False):
        # フラグをリセット（1回だけ更新するため）
        st.session_state['auto_reflect_to_sidebar'] = False
    # 計算設定
    st.sidebar.title('計算設定')
    
    # 選択値を取得（セッションステートには保存しない）
    basis_set = st.sidebar.selectbox('基底関数セット:', ['sto-3g', '3-21g', '6-31g', 'cc-pvdz', 'cc-pvtz'], index=0)
    functional = st.sidebar.selectbox('交換相関汎関数:', ['b3lyp', 'pbe', 'pbe0', 'lda', 'm06'], index=0)
    charge = st.sidebar.number_input('分子の電荷:', min_value=-5, max_value=5, value=0, step=1)
    spin = st.sidebar.number_input('スピン多重度 (2S+1):', min_value=1, max_value=6, value=1, step=1)
    
    return {
        'input_method': input_method,
        'pubchem_query': pubchem_query,
        'smiles': smiles,
        'basis_set': basis_set,
        'functional': functional,
        'charge': charge,
        'spin': spin,
        'xyz_string': xyz_string  # ここでサイドバーのテキストエリアの値を返す
    }
