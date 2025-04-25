import plotly.graph_objects as go
import py3Dmol
import tempfile
import os
from pyscf import tools

def plot_mo_energies_plotly(orbital_energies, homo_idx, gap=None):
    """分子軌道エネルギー準位図をPlotlyで描画"""
    df = orbital_energies.copy()
    df['タイプ'] = 'MO'
    df.at[homo_idx, 'タイプ'] = 'HOMO'
    df.at[homo_idx+1, 'タイプ'] = 'LUMO'
    colors = {'MO': 'gray', 'HOMO': 'blue', 'LUMO': 'red'}
    fig = go.Figure()
    # HOMO/LUMOを太く、他は細く
    for i, row in df.iterrows():
        color = colors.get(row['タイプ'], 'gray')
        width = 6 if row['タイプ'] in ['HOMO', 'LUMO'] else 2
        fig.add_shape(type="line",
            x0=0.3, x1=0.7,
            y0=row['エネルギー (eV)'], y1=row['エネルギー (eV)'],
            line=dict(color=color, width=width),
            layer="above"
        )
        fig.add_trace(go.Scatter(
            x=[0.5], y=[row['エネルギー (eV)']],
            mode='markers+text',
            marker=dict(color=color, size=12 if row['タイプ'] in ['HOMO','LUMO'] else 7),
            text=[row['タイプ'] if row['タイプ'] in ['HOMO','LUMO'] else ''],
            textposition="middle right",
            showlegend=False,
            hoverinfo='text',
            hovertext=f"{row['タイプ']}<br>軌道番号: {row['軌道番号']}<br>エネルギー: {row['エネルギー (eV)']:.4f} eV"
        ))
    # HOMO-LUMO gap矢印
    homo_e = df.at[homo_idx, 'エネルギー (eV)']
    lumo_e = df.at[homo_idx+1, 'エネルギー (eV)']
    gap_value = gap if gap is not None else lumo_e - homo_e
    fig.add_annotation(
        x=0.85, y=(homo_e + lumo_e)/2,
        ax=0.85, ay=homo_e,
        xref='x', yref='y', axref='x', ayref='y',
        text=f"GAP: {gap_value:.2f} eV",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="green",
        font=dict(color="green", size=14),
        align="center",
        bgcolor="white",
        bordercolor="green"
    )
    # HOMO/LUMOラベル
    fig.add_annotation(x=0.9, y=homo_e, text=f"HOMO: {homo_e:.2f} eV", showarrow=False, font=dict(color="blue", size=13), bgcolor="white")
    fig.add_annotation(x=0.9, y=lumo_e, text=f"LUMO: {lumo_e:.2f} eV", showarrow=False, font=dict(color="red", size=13), bgcolor="white")
    fig.update_layout(
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
        yaxis=dict(title='エネルギー (eV)', zeroline=True, showgrid=True, gridcolor='lightgray'),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor='white',
    )
    return fig

def visualize_molecular_orbital(mol, mf, orbital_index, orbital_name):
    """分子軌道を3D可視化"""
    result = {}
    try:
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
        
        result['view'] = view
        result['xyz_block'] = xyz_block
        result['success'] = True
        return result
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        return result

def visualize_molecule_3d(atoms, coords):
    """分子の3D構造を表示"""
    if atoms is None or coords is None:
        return None
    
    xyz_block = f"{len(atoms)}\n\n" + '\n'.join([
        f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" for a, c in zip(atoms, coords)
    ])
    view = py3Dmol.view(width=max, height=350)
    view.addModel(xyz_block, 'xyz')
    view.setStyle({'stick': {}})
    view.zoomTo()
    return view
