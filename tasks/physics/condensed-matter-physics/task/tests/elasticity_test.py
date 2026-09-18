import json, math
from pathlib import Path
import numpy as np
import pandas as pd

DATA=Path('/app/data/DFT_ENERGY_STRAINS.csv')
POSCAR=Path('/app/data/POSCAR_UNITCELL')
OUTPUT=Path('/app/output/elastic_properties.json')
M={'Mg':24.305,'Si':28.085,'O':15.999}
EVJ=1.602176634e-19; AMU=1.66053906660e-27; A3=1e-30
EXPECTED={'e1','e2','e3','e4','e5','e6','e1_e2','e1_e3','e2_e3'}

def require_output():
    assert OUTPUT.exists(), 'solution did not create /app/output/elastic_properties.json'
    return json.loads(OUTPUT.read_text())

def ref_structure():
    lines=[x.strip() for x in POSCAR.read_text().splitlines() if x.strip()]
    s=float(lines[1]); raw=np.array([[float(v) for v in lines[i].split()[:3]] for i in range(2,5)])
    L=raw*s if s>0 else raw*(abs(s)/abs(np.linalg.det(raw)))**(1/3)
    sp=lines[5].split(); ct=[int(x) for x in lines[6].split()]
    V=abs(np.linalg.det(L)); mass=sum(M[x]*n for x,n in zip(sp,ct)); rho=mass*AMU/(V*A3)
    return V,rho,sum(ct)

def independent_quartic_curvature(x,y):
    # Independent formulation: normalized Vandermonde least-squares, not np.polyfit.
    scale=max(np.max(np.abs(x)),1.0); z=x/scale
    A=np.column_stack([np.ones_like(z),z,z**2,z**3,z**4])
    coef=np.linalg.lstsq(A,y,rcond=None)[0]
    return 2*coef[2]/scale**2

def ref_C():
    V,_,_=ref_structure(); d=pd.read_csv(DATA)
    curv={p:independent_quartic_curvature(g.eta.to_numpy(float),g.energy_ev.to_numpy(float)) for p,g in d.groupby('strain_pattern')}
    f=(EVJ/A3)/(1e9*V); C=np.zeros((6,6))
    for i in range(6): C[i,i]=curv[f'e{i+1}']*f
    C[0,1]=C[1,0]=0.5*(curv['e1_e2']-curv['e1']-curv['e2'])*f
    C[0,2]=C[2,0]=0.5*(curv['e1_e3']-curv['e1']-curv['e3'])*f
    C[1,2]=C[2,1]=0.5*(curv['e2_e3']-curv['e2']-curv['e3'])*f
    return C

def ref_vrh(C):
    S=np.linalg.inv(C); a,b,c=C[0,0],C[1,1],C[2,2]; d,e,f=C[0,1],C[0,2],C[1,2]; g,h,i=C[3,3],C[4,4],C[5,5]
    BV=(a+b+c+2*(d+e+f))/9; GV=(a+b+c-d-e-f+3*(g+h+i))/15
    BR=1/(S[0,0]+S[1,1]+S[2,2]+2*(S[0,1]+S[0,2]+S[1,2]))
    GR=15/(4*(S[0,0]+S[1,1]+S[2,2])-4*(S[0,1]+S[0,2]+S[1,2])+3*(S[3,3]+S[4,4]+S[5,5]))
    B=(BV+BR)/2; G=(GV+GR)/2
    return B,G,9*B*G/(3*B+G),(3*B-2*G)/(2*(3*B+G)),5*GV/GR+BV/BR-6,BV,BR,GV,GR

def test_output_exists_and_json():
    r=require_output(); assert isinstance(r,dict)

def test_structure_independent():
    r=require_output(); V,rho,n=ref_structure()
    assert math.isclose(r['V0_A3'],V,rel_tol=1e-10,abs_tol=1e-10)
    assert math.isclose(r['density_kg_m3'],rho,rel_tol=1e-10,abs_tol=1e-8)
    assert r['num_atoms']==n

def test_full_stiffness_independent():
    r=require_output(); C=ref_C()
    assert np.asarray(r['C_ij']).shape==(6,6)
    assert np.allclose(r['C_ij'],C,rtol=3e-5,atol=3e-5)

def test_compliance_is_inverse():
    r=require_output(); C=np.asarray(r['C_ij'],float); S=np.asarray(r['S_ij'],float)
    assert np.allclose(C,C.T,atol=1e-9); assert np.allclose(S,S.T,atol=1e-9); assert np.allclose(C@S,np.eye(6),atol=2e-7)

def test_stability():
    r=require_output(); C=np.asarray(r['C_ij']); eig=np.linalg.eigvalsh(C)
    assert np.all(eig>0)
    assert all(bool(v) for v in r['Born_stability'].values())

def test_vrh_independent():
    r=require_output(); q=ref_vrh(ref_C()); names=['B_VRH','G_VRH','E_VRH','nu_VRH','Anisotropy_AU']
    for k,v in zip(names,q[:5]): assert math.isclose(r[k],v,rel_tol=4e-5,abs_tol=4e-7)

def test_voigt_reuss_bounds():
    r=require_output();
    for x in ['B','G']:
        v=r[x+'_V']; rr=r[x+'_R']; h=r[x+'_VRH']; assert min(v,rr)-1e-8 <= h <= max(v,rr)+1e-8

def test_fit_diagnostics():
    r=require_output(); fd=r['fit_diagnostics']; assert set(fd)==EXPECTED
    for x in fd.values():
        assert x['R2']>0.99; assert x['rms_residual_eV']<0.002; assert x['leave_one_out_curvature_std_eV']>=0

def test_uncertainty_reproducible_contract():
    r=require_output(); b=r['bootstrap_uncertainty']; assert b['seed']==20260910; assert b['n_samples']>=150
    assert set(b['std'])=={'C11','C22','C33','C44','C55','C66','B_VRH','G_VRH','E_VRH','nu_VRH','Anisotropy_AU'}
    assert all(np.isfinite(list(b['std'].values())))

def test_directional_elasticity():
    r=require_output(); d=r['directional'];
    assert d['E_min_GPa']>0; assert d['E_max_GPa']>=d['E_min_GPa']; assert d['E_anisotropy']>=1
    for k in ['E_min_direction','E_max_direction']:
        n=np.asarray(d[k]); assert np.isclose(np.linalg.norm(n),1,atol=2e-6)

def test_acoustic_physics():
    r=require_output();
    for k in ['v_l','v_t','v_m','Debye_temperature']:
        assert np.isfinite(r[k]) and r[k]>0
    assert r['v_l']>r['v_t']>0
    a=r['acoustic_anisotropy']; assert a['directional_min_velocity_m_s']>0; assert a['directional_max_velocity_m_s']>=a['directional_min_velocity_m_s']

def test_required_schema():
    r=require_output(); req={'V0_A3','density_kg_m3','num_atoms','C_ij','S_ij','B_VRH','G_VRH','E_VRH','nu_VRH','Pugh_ratio','Anisotropy_AU','v_l','v_t','v_m','Debye_temperature','directional','acoustic_anisotropy','fit_diagnostics','bootstrap_uncertainty'}
    assert req.issubset(r)

def test_no_hardcoded_nan_inf():
    r=require_output()
    vals=[r[k] for k in ['V0_A3','density_kg_m3','B_VRH','G_VRH','E_VRH','nu_VRH','Pugh_ratio','Anisotropy_AU','v_l','v_t','v_m','Debye_temperature']]
    assert all(np.isfinite(vals))
