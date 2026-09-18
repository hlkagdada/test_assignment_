import json, math, os, warnings
from numpy.polynomial import Polynomial
from pathlib import Path
import numpy as np
import pandas as pd

DATA_CSV='/app/data/DFT_ENERGY_STRAINS.csv'; POSCAR_PATH='/app/data/POSCAR_UNITCELL'; OUTPUT_DIR='/app/output'
EV_TO_J=1.602176634e-19; ANG3_TO_M3=1e-30; AMU_TO_KG=1.66053906660e-27; GPA_TO_PA=1e9
MASSES={'Mg':24.305,'Si':28.085,'O':15.999}
EXPECTED={'e1','e2','e3','e4','e5','e6','e1_e2','e1_e3','e2_e3'}

def read_poscar(path):
    lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    scale=float(lines[1]); raw=np.array([[float(v) for v in lines[i].split()[:3]] for i in range(2,5)],float)
    if scale>0: lattice=scale*raw
    elif scale<0: lattice=raw*(abs(scale)/abs(np.linalg.det(raw)))**(1/3)
    else: raise ValueError('zero POSCAR scale')
    species=lines[5].split(); counts=[int(x) for x in lines[6].split()]
    if len(species)!=len(counts): raise ValueError('species/count mismatch')
    return lattice,species,counts

def structure(lattice,species,counts):
    volume=abs(np.linalg.det(lattice)); norms=np.linalg.norm(lattice,axis=1)
    cos=np.array([[np.dot(lattice[i],lattice[j])/(norms[i]*norms[j]) for j in range(3)] for i in range(3)])
    ortho_err=float(np.max(np.abs(cos-np.eye(3))))
    mass_amu=sum(MASSES[s]*n for s,n in zip(species,counts))
    rho=mass_amu*AMU_TO_KG/(volume*ANG3_TO_M3)
    return volume,rho,ortho_err,mass_amu,sum(counts)

def load_data(path):
    df=pd.read_csv(path); req={'strain_pattern','eta','energy_ev'}
    if not req.issubset(df.columns): raise ValueError(f'missing columns: {sorted(req-set(df.columns))}')
    if set(df.strain_pattern)!=EXPECTED: raise ValueError(f'strain patterns mismatch: {sorted(EXPECTED-set(df.strain_pattern))}')
    if not np.isfinite(df[['eta','energy_ev']].to_numpy()).all(): raise ValueError('non-finite data')
    if df.duplicated(['strain_pattern','eta']).any(): raise ValueError('duplicate strain point')
    return df

def fit_one(x,y):
    if len(x)<5: raise ValueError('need >=5 points')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.exceptions.RankWarning if hasattr(np, 'exceptions') else np.RankWarning)
        c=np.polyfit(x,y,4)
    pred=np.polyval(c,x); resid=pred-y
    ssr=float(np.sum(resid**2)); sst=float(np.sum((y-y.mean())**2))
    r2=1-ssr/sst if sst else 1.0
    return c,2*c[2],float(np.sqrt(np.mean(resid**2))),float(r2),float(c[3])

def diagnostics(df):
    out={}; curv={}; coeff={}
    rng=np.random.default_rng(12345)
    boot_curv={p:[] for p in EXPECTED}
    for p,g in df.groupby('strain_pattern'):
        x=g.eta.to_numpy(float); y=g.energy_ev.to_numpy(float); c,k,rms,r2,a1=fit_one(x,y)
        coeff[p]=c.tolist(); curv[p]=float(k)
        # symmetry residual: compare E(+eta) and E(-eta) where both exist
        mp={round(float(a),12):float(b) for a,b in zip(x,y)}; pairs=[]
        for a,b in mp.items():
            if a>0 and round(-a,12) in mp: pairs.append(abs(b-mp[round(-a,12)]))
        sym=float(max(pairs)) if pairs else 0.0
        loo=[]
        for i in range(len(x)):
            mask=np.ones(len(x),bool); mask[i]=False
            loo.append(fit_one(x[mask],y[mask])[1])
        loo=np.asarray(loo)
        # deterministic bootstrap curvature ensemble
        for _ in range(200):
            idx=rng.integers(0,len(x),len(x)); boot_curv[p].append(fit_one(x[idx],y[idx])[1])
        out[p]={'coefficients':c.tolist(),'curvature_eV':float(k),'rms_residual_eV':rms,'R2':r2,'linear_coefficient_eV':a1,
                'max_pm_symmetry_residual_eV':sym,'leave_one_out_curvature_std_eV':float(np.std(loo,ddof=1)),
                'bootstrap_curvature_std_eV':float(np.std(boot_curv[p],ddof=1))}
    return curv,coeff,out,boot_curv

def make_C(k,V):
    f=(EV_TO_J/ANG3_TO_M3)/GPA_TO_PA/V
    c11,c22,c33=[k[f'e{i}']*f for i in (1,2,3)]; c44,c55,c66=[k[f'e{i}']*f for i in (4,5,6)]
    c12=.5*(k['e1_e2']-k['e1']-k['e2'])*f; c13=.5*(k['e1_e3']-k['e1']-k['e3'])*f; c23=.5*(k['e2_e3']-k['e2']-k['e3'])*f
    return np.array([[c11,c12,c13,0,0,0],[c12,c22,c23,0,0,0],[c13,c23,c33,0,0,0],[0,0,0,c44,0,0],[0,0,0,0,c55,0],[0,0,0,0,0,c66]],float)

def born(C):
    c11,c22,c33=C[0,0],C[1,1],C[2,2]; c12,c13,c23=C[0,1],C[0,2],C[1,2]; c44,c55,c66=C[3,3],C[4,4],C[5,5]
    checks={'C44>0':bool(c44>0),'C55>0':bool(c55>0),'C66>0':bool(c66>0),'C11>abs(C12)':bool(c11>abs(c12)),'C22>abs(C12)':bool(c22>abs(c12)),'C33>abs(C13)':bool(c33>abs(c13)),'C11+C22+C33+2(C12+C13+C23)>0':bool(c11+c22+c33+2*(c12+c13+c23)>0)}
    return checks

def vrh(C):
    S=np.linalg.inv(C); a,b,c=C[0,0],C[1,1],C[2,2]; d,e,f=C[0,1],C[0,2],C[1,2]; g,h,i=C[3,3],C[4,4],C[5,5]
    BV=(a+b+c+2*(d+e+f))/9; GV=(a+b+c-d-e-f+3*(g+h+i))/15
    BR=1/(S[0,0]+S[1,1]+S[2,2]+2*(S[0,1]+S[0,2]+S[1,2]))
    GR=15/(4*(S[0,0]+S[1,1]+S[2,2])-4*(S[0,1]+S[0,2]+S[1,2])+3*(S[3,3]+S[4,4]+S[5,5]))
    B=(BV+BR)/2; G=(GV+GR)/2; E=9*B*G/(3*B+G); nu=(3*B-2*G)/(2*(3*B+G)); AU=5*GV/GR+BV/BR-6
    return {'S':S,'B_V':BV,'B_R':BR,'B_VRH':B,'G_V':GV,'G_R':GR,'G_VRH':G,'E_VRH':E,'nu_VRH':nu,'Pugh_ratio':B/G,'Anisotropy_AU':AU,
            'B_spread':BV-BR,'G_spread':GV-GR}

def directions(n_theta=91,n_phi=180):
    th=np.linspace(0,np.pi,n_theta); ph=np.linspace(0,2*np.pi,n_phi,endpoint=False); arr=[]
    for t in th:
        for p in ph: arr.append((np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)))
    return np.asarray(arr)

def directional(C,S):
    n=directions(); l,m,z=n.T
    invE=S[0,0]*l**4+S[1,1]*m**4+S[2,2]*z**4+(2*S[0,1]+S[5,5])*l*l*m*m+(2*S[0,2]+S[4,4])*l*l*z*z+(2*S[1,2]+S[3,3])*m*m*z*z
    E=1/invE
    beta=(S[0,0]+S[0,1]+S[0,2])*l*l+(S[0,1]+S[1,1]+S[1,2])*m*m+(S[0,2]+S[1,2]+S[2,2])*z*z
    ibeta=np.argmin(beta); abeta=np.argmax(beta); ie=np.argmin(E); ae=np.argmax(E)
    return {'E_min_GPa':float(E[ie]),'E_max_GPa':float(E[ae]),'E_anisotropy':float(E[ae]/E[ie]),'E_min_direction':n[ie].tolist(),'E_max_direction':n[ae].tolist(),
            'linear_compressibility_min_GPa_inv':float(beta[ibeta]),'linear_compressibility_max_GPa_inv':float(beta[abeta]),'linear_compressibility_min_direction':n[ibeta].tolist(),'linear_compressibility_max_direction':n[abeta].tolist()}

def christoffel(C,n):
    # tensor C_ijkl for orthorhombic, with engineering shear convention
    T=np.zeros((3,3,3,3));
    for i,j in [(0,0),(1,1),(2,2)]:
        for k,l in [(0,0),(1,1),(2,2)]:
            vo={(0,0):0,(1,1):1,(2,2):2}; T[i,j,k,l]=C[vo[(i,j)],vo[(k,l)]]
    for (a,b,idx) in [(1,2,3),(0,2,4),(0,1,5)]:
        q=C[idx,idx]
        for ij in [(a,b),(b,a)]:
            for kl in [(a,b),(b,a)]: T[ij[0],ij[1],kl[0],kl[1]]=q
    return np.einsum('ijkl,j,l->ik',T,n,n)

def acoustic(C,B,G,rho):
    n=directions(61,120); vals=[]
    for v in n:
        ev=np.linalg.eigvalsh(christoffel(C,v)); vel=np.sqrt(np.maximum(ev,0)*GPA_TO_PA/rho); vals.append(vel)
    vals=np.asarray(vals); idx=np.unravel_index(np.argmin(vals),vals.shape); jdx=np.unravel_index(np.argmax(vals),vals.shape)
    vl=math.sqrt((B+4*G/3)*GPA_TO_PA/rho); vt=math.sqrt(G*GPA_TO_PA/rho); vm=((1/3)*(vl**-3+2*vt**-3))**(-1/3)
    return {'v_l':vl,'v_t':vt,'v_m':vm,'Debye_temperature':(6.62607015e-34/1.380649e-23)*vm*((3*sum(read_poscar(POSCAR_PATH)[2])/(4*np.pi*(abs(np.linalg.det(read_poscar(POSCAR_PATH)[0]))*ANG3_TO_M3))))**(1/3),
            'directional_min_velocity_m_s':float(vals[idx]),'directional_max_velocity_m_s':float(vals[jdx]),'directional_min_velocity_direction':n[idx[0]].tolist(),'directional_max_velocity_direction':n[jdx[0]].tolist()}

def bootstrap_properties(df,V):
    rng=np.random.default_rng(20260910); samples=[]
    groups={p:g for p,g in df.groupby('strain_pattern')}
    for _ in range(200):
        k={}
        for p,g in groups.items():
            x=g.eta.to_numpy(float); y=g.energy_ev.to_numpy(float); idx=rng.integers(0,len(x),len(x)); k[p]=fit_one(x[idx],y[idx])[1]
        C=make_C(k,V)
        try:
            q=vrh(C); samples.append([*C[0:3,0:3].diagonal(),C[3,3],C[4,4],C[5,5],q['B_VRH'],q['G_VRH'],q['E_VRH'],q['nu_VRH'],q['Anisotropy_AU']])
        except np.linalg.LinAlgError: pass
    a=np.asarray(samples); keys=['C11','C22','C33','C44','C55','C66','B_VRH','G_VRH','E_VRH','nu_VRH','Anisotropy_AU']
    std=dict(zip(keys,np.std(a,axis=0,ddof=1))); means=dict(zip(keys,np.mean(a,axis=0))); rel={k:(std[k]/abs(means[k]) if means[k] else float('inf')) for k in keys}
    return {'n_samples':int(len(a)),'seed':20260910,'std':std,'mean':means,'relative_std':rel,'largest_relative_uncertainty':max(rel,key=rel.get)}

def main():
    os.makedirs(OUTPUT_DIR,exist_ok=True); lattice,species,counts=read_poscar(POSCAR_PATH); V,rho,ortho,mass,natoms=structure(lattice,species,counts); df=load_data(DATA_CSV)
    curv,coeff,fd,bc=diagnostics(df); C=make_C(curv,V); Sprops=vrh(C); eig=np.linalg.eigvalsh(C); checks=born(C)
    if ortho>1e-8: raise ValueError('lattice is not orthorhombic');
    if not np.all(eig>0): raise ValueError('stiffness matrix is not positive definite')
    if not all(checks.values()): raise ValueError(f'Born stability failure: {checks}')
    d=directional(C,Sprops['S']); ac=acoustic(C,Sprops['B_VRH'],Sprops['G_VRH'],rho); bu=bootstrap_properties(df,V)
    result={'V0_A3':V,'density_kg_m3':rho,'num_atoms':natoms,'composition':dict(zip(species,counts)),'cell_mass_amu':mass,'orthorhombic_error':ortho,
      'C_ij':C.tolist(),'S_ij':Sprops['S'].tolist(),'C_eigenvalues_GPa':eig.tolist(),'Born_stability':checks,'fit_curvatures_eV':curv,'fit_coefficients':coeff,'fit_diagnostics':fd,
      'B_V':Sprops['B_V'],'B_R':Sprops['B_R'],'B_VRH':Sprops['B_VRH'],'G_V':Sprops['G_V'],'G_R':Sprops['G_R'],'G_VRH':Sprops['G_VRH'],'E_VRH':Sprops['E_VRH'],'nu_VRH':Sprops['nu_VRH'],'Pugh_ratio':Sprops['Pugh_ratio'],'Anisotropy_AU':Sprops['Anisotropy_AU'],'B_spread':Sprops['B_spread'],'G_spread':Sprops['G_spread'],'directional':d,'acoustic_anisotropy':ac,'v_l':ac['v_l'],'v_t':ac['v_t'],'v_m':ac['v_m'],'Debye_temperature':ac['Debye_temperature'],'bootstrap_uncertainty':bu}
    # finite-value guard
    def finite(x): return np.isfinite(x) if isinstance(x,(int,float,np.floating)) else True
    if not all(finite(x) for x in [V,rho,*eig]): raise ValueError('non-finite output')
    Path(OUTPUT_DIR,'elastic_properties.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
