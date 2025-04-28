import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 溶媒の誘電率定数
SOLVENT_DIELECTRIC_CONSTANTS = {
    "Water": 78.3553,
    "Dimethylsulfoxide": 46.826,
    "N,N-Dimethylformamide": 37.219,
    "Nitromethane": 36.562,
    "Methanol": 32.613,
    "Ethanol": 24.852,
    "Acetone": 20.493,
    "Dichloroethane": 10.125,
    "Dichloromethane": 8.93,
    "Tetrahydrofuran": 7.4297,
    "Chlorobenzene": 5.6968,
    "Chloroform": 4.7113,
    "Diethylether": 4.2400,
    "Toluene": 2.3741,
    "Benzene": 2.2706,
    "1,4-Dioxane": 2.2099,
    "Cyclohexane": 2.0160
}

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
        # 溶媒情報の取得
        solvent_info = ""
        if st.session_state.get('enable_solvent', False):
            solvent_model = st.session_state.get('solvent_model', '')
            selected_solvent = st.session_state.get('selected_solvent', 'カスタム')
            
            if solvent_model == 'IEF-PCM':
                if selected_solvent == 'カスタム':
                    epsilon = st.session_state.get('custom_epsilon', 0.0)
                    if epsilon is not None:
                        solvent_info = f"- 溶媒効果: **{solvent_model}** (カスタム, ε={epsilon:.4f})"
                    else:
                        solvent_info = f"- 溶媒効果: **{solvent_model}** (カスタム)"
                else:
                    solvent_info = f"- 溶媒効果: **{solvent_model}** ({selected_solvent})"
            elif solvent_model == 'SMD':
                if selected_solvent == 'カスタム':
                    solvent_info = f"- 溶媒効果: **{solvent_model}** (Water)"
                else:
                    solvent_info = f"- 溶媒効果: **{solvent_model}** ({selected_solvent})"
        
        st.sidebar.info(f"""
        **現在のDFT計算設定:**
        - 基底関数セット: **{st.session_state.get('basis_set', 'sto-3g')}**
        - 交換相関汎関数: **{st.session_state.get('functional', 'b3lyp')}**
        - 電荷: {st.session_state.get('charge', 0)}
        - スピン多重度: {st.session_state.get('spin', 1)}
        {solvent_info}
        """)
    
    st.sidebar.title('分子構造入力')
    input_method = st.sidebar.radio('**入力方法**', ['PubChem名/IDから取得', 'SMILESから取得'], index=0, key='input_method_radio_mol')
    pubchem_query = ''
    smiles = ''
    xyz_string = st.session_state.get('xyz_string', '')
    if input_method == 'PubChem名/IDから取得':
        pubchem_query = st.sidebar.text_input('化合物名またはCID', key='pubchem_sidebar_input')
    else:
        smiles = st.sidebar.text_input('SMILES', key='smiles_sidebar_input')
    # 変換ボタンを一番下に
    convert_btn = st.sidebar.button('xyzに変換', key='convert_btn_xyz')
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
    xyz_input_key = 'xyz_input_sidebar_main'
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
    
    # 計算負荷に基づく基底関数と汎関数のリスト（色分け表示用）
    st.sidebar.markdown("計算負荷の目安:")
    st.sidebar.markdown('''
    <span style="color:green;">■ 軽い</span> | 
    <span style="color:#CC9900;">■ 中程度</span> | 
    <span style="color:orange;">■ 重い</span> | 
    <span style="color:red;">■ 非常に重い</span>
    ''', unsafe_allow_html=True)
    
    # 基底関数のオプションを計算負荷で色分け
    basis_options = [
        {'label': '<span style="color:green;">sto-3g</span>', 'value': 'sto-3g'},
        {'label': '<span style="color:#CC9900;">3-21g</span>', 'value': '3-21g'},
        {'label': '<span style="color:#CC9900;">6-31g</span>', 'value': '6-31g'},
        {'label': '<span style="color:orange;">6-31g(d)</span>', 'value': '6-31g(d)'},
        {'label': '<span style="color:orange;">6-31+g(d)</span>', 'value': '6-31+g(d)'},
        {'label': '<span style="color:red;">6-31+g(d,p)</span>', 'value': '6-31+g(d,p)'},
        {'label': '<span style="color:red;">cc-pvdz</span>', 'value': 'cc-pvdz'},
        {'label': '<span style="color:red;">cc-pvtz</span>', 'value': 'cc-pvtz'}
    ]
    
    # 色付きラベルの表示
    basis_labels = [opt['label'] for opt in basis_options]
    basis_values = [opt['value'] for opt in basis_options]
    
    # 基底関数の選択肢を表示
    basis_index = st.sidebar.selectbox(
        '基底関数セット:', 
        range(len(basis_options)),
        format_func=lambda i: basis_options[i]['value'],
        index=0, 
        key='basis_set_select_main'
    )
    basis_set = basis_values[basis_index]
    
    # 選択された基底関数を色付きで表示
    st.sidebar.markdown(f'選択: {basis_labels[basis_index]}', unsafe_allow_html=True)
    
    # 汎関数の設定    
    # 汎関数のオプションを計算負荷で色分け
    functional_options = [
        {'label': '<span style="color:green;">lda</span>', 'value': 'lda'},
        {'label': '<span style="color:#CC9900;">pbe</span>', 'value': 'pbe'},
        {'label': '<span style="color:orange;">pbe0</span>', 'value': 'pbe0'},
        {'label': '<span style="color:orange;">b3lyp</span>', 'value': 'b3lyp'},
        {'label': '<span style="color:red;">m06</span>', 'value': 'm06'},
        {'label': '<span style="color:red;">wb97x</span>', 'value': 'wb97x'},
        {'label': '<span style="color:red;">cam-b3lyp</span>', 'value': 'cam-b3lyp'}
    ]
    
    # 色付きラベルの表示
    functional_labels = [opt['label'] for opt in functional_options]
    functional_values = [opt['value'] for opt in functional_options]
    
    # 汎関数の選択肢を表示
    functional_index = st.sidebar.selectbox(
        '交換相関汎関数:', 
        range(len(functional_options)),
        format_func=lambda i: functional_options[i]['value'],
        index=3,  # b3lypをデフォルトに
        key='functional_select_main'
    )
    functional = functional_values[functional_index]
    
    # 選択された汎関数を色付きで表示
    st.sidebar.markdown(f'選択: {functional_labels[functional_index]}', unsafe_allow_html=True)
    charge = st.sidebar.number_input('分子の電荷:', min_value=-5, max_value=5, value=0, step=1, key='charge_input_main')
    spin = st.sidebar.number_input('スピン多重度 (2S+1):', min_value=1, max_value=6, value=1, step=1, key='spin_input_main')
    
    # 溶媒効果の設定
    # 溶媒モデルの選択
    solvent_model = st.sidebar.selectbox(
        '溶媒効果:',
        ['なし', 'IEF-PCM', 'SMD'],
        index=0,
        key='solvent_model_select_main'
    )
    
    # 溶媒効果が有効かどうか
    enable_solvent = solvent_model != 'なし'
    st.session_state['enable_solvent'] = enable_solvent
    st.session_state['solvent_model'] = solvent_model
    
    # 溶媒効果の設定結果を格納する辞書
    solvent_settings = {
        'enable_solvent': enable_solvent,
        'solvent_model': solvent_model,
        'selected_solvent': None,
        'epsilon': None
    }
    
    if enable_solvent:
        # 溶媒の選択
        solvent_options = ["カスタム"] + list(SOLVENT_DIELECTRIC_CONSTANTS.keys())
        selected_solvent = st.sidebar.selectbox(
            '溶媒:',
            solvent_options,
            index=0,
            key='solvent_select_main'
        )
        st.session_state['selected_solvent'] = selected_solvent
        solvent_settings['selected_solvent'] = selected_solvent
        
        # カスタム誘電率の入力（IEF-PCMのみ）
        epsilon = None
        if solvent_model == 'IEF-PCM':
            if selected_solvent == "カスタム":
                custom_epsilon = st.sidebar.number_input(
                    '誘電率（ε）:',
                    min_value=1.0,
                    max_value=100.0,
                    value=st.session_state.get('custom_epsilon', 78.3553),
                    step=0.1,
                    format="%.4f",
                    help="溶媒の誘電率を直接指定します。水の誘電率は約78.3553です。",
                    key='custom_epsilon_input_main'
                )
                st.session_state['custom_epsilon'] = custom_epsilon
                epsilon = custom_epsilon
                solvent_settings['epsilon'] = epsilon
            else:
                epsilon = SOLVENT_DIELECTRIC_CONSTANTS.get(selected_solvent, 78.3553)
                solvent_settings['epsilon'] = epsilon
                st.sidebar.info(f'{selected_solvent}の誘電率: {epsilon:.4f}')
        
        # 溶媒効果の説明
        with st.sidebar.expander("ℹ️ 溶媒効果について", expanded=False):
            st.markdown("""
            ### 溶媒効果とは
            
            分子が溶媒中に存在する場合、その電子構造は真空中と比較して変化します。これが溶媒効果です。
            
            ### 溶媒モデル
            
            - **IEF-PCM (Polarizable Continuum Model)**: 溶媒を連続した誘電体として扱います。溶質分子の周りに形成される静電場を計算します。
            
            - **SMD (Solvation Model Density)**: PCMを拡張したモデルで、非電解質の溶媒和をより正確に表現します。表面張力や溶媒の分子構造も考慮します。
            
            ### 誘電率（ε）
            
            誘電率は溶媒の極性を表す重要なパラメータです。値が大きいほど極性が高く、小さいほど非極性です。
            
            - 水（高極性）: ε ≈ 78.4
            - アルコール（中極性）: ε ≈ 20-35
            - クロロホルム（低極性）: ε ≈ 4.7
            - ヘキサン（非極性）: ε ≈ 1.9
            """)
    
    # 開発者向け設定（展開可能なセクション）
    with st.sidebar.expander("⚙️ 開発者向け設定", expanded=False):
        st.markdown("""
        ### 高度な計算設定
        以下の設定は計算パフォーマンスに大きく向上させますが、環境により動作が不安定になる場合があります。不具合が発生した場合は、デフォルト値に戻してお試しください。OS版では標準提供予定です。
        """)
        st.markdown("---")
        
        # 並列計算設定
        available_cores = st.session_state.get('available_cpu_cores', 1)
        
        # CPUコア数の設定（最小1コア、最大は利用可能コア数）
        cpu_cores = st.slider(
            'CPU使用コア数:',
            min_value=1,
            max_value=available_cores,
            value=st.session_state.get('num_cpu_cores', 1),
            step=1,
            help='CPU設定ガイド\n- コア数を増やすと計算速度が向上しますが、メモリ消費も増加します\n- 大きな分子では少ないコア数が安定します（1〜4コア推奨）\n- 小さな分子では多くのコアを活用できます',
            key='cpu_cores_slider_main'
        )
        
        # ガイダンス表示
        st.markdown("""
        **[動作環境について]**
        - この機能はDockerコンテナ内での実行が想定されています。streamlit Cloud上では動作保証しません。
        - cloneしたリポジトリをDockerを使用してローカルで実行してください。
        - `compose.yaml`ファイルの`cpus: '8'`の設定により最大8コアまで使用可能です
        - メモリ制限は`memory: 8G`で設定されています(8GのRAMのPCを使っている人は設定を変更してください)
        
        **[トラブルシューティング]**
        - **エラー発生時**: 計算エラーが発生した場合は、コア数を1に戻してください
        - **Out of Memory**: メモリ不足エラーの場合もコア数を減らしてください
        - **Docker Desktop側のエラー**: 初期設定だとDocker Decktop側の設定でCPUコア数やメモリ数に制限がかかっている場合があります。その場合は先に緩和してください。
        """)
    
    # セッションに保存
    st.session_state['num_cpu_cores'] = cpu_cores
    
    return {
        'input_method': input_method,
        'pubchem_query': pubchem_query,
        'smiles': smiles,
        'basis_set': basis_set,
        'functional': functional,
        'charge': charge,
        'spin': spin,
        'cpu_cores': cpu_cores,  # CPU設定を追加
        'xyz_string': xyz_string,  # ここでサイドバーのテキストエリアの値を返す
        'solvent_settings': solvent_settings  # 溶媒効果の設定を追加
    }
