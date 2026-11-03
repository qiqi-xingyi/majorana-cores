"""Charge/degree diagnostics."""
from __future__ import annotations
import itertools, math
import numpy as np
from scipy import sparse, linalg

def clifford_product(left,right):
    result={}
    for a,ca in left.items():
        for b,cb in right.items():
            out=list(a); sign=1
            for j in b:
                sign*=(-1)**sum(x>j for x in out)
                if j in out:out.remove(j)
                else:out.append(j);out.sort()
            key=tuple(out);result[key]=result.get(key,0)+sign*ca*cb
    return {k:v for k,v in result.items() if v!=0}

def combine(*items):
    out={}
    for scale,p in items:
        for k,v in p.items():out[k]=out.get(k,0)+scale*v
    return {k:v for k,v in out.items() if v!=0}

def creation(p):
    return {(2*p,):1/math.sqrt(2),(2*p+1,):-1j/math.sqrt(2)}

def annihilation(p):
    return {k:np.conj(v) for k,v in creation(p).items()}

def phase_degree(d):return (1j)**(d*(d-1)//2)

def local_basis(q):
    labels={d:list(itertools.combinations(range(2*q),d)) for d in [1,2,3]}
    indices={d:{s:i for i,s in enumerate(v)} for d,v in labels.items()}
    def matrix(polys,degree):
        rr=[];cc=[];vv=[]
        for j,poly in enumerate(polys):
            for key,v in poly.items():
                if len(key)!=degree:raise AssertionError('Dictionary has mixed degree')
                rr.append(indices[degree][key]);cc.append(j);vv.append(v/phase_degree(degree))
        return sparse.csr_matrix((vv,(rr,cc)),shape=(len(labels[degree]),len(polys)),dtype=complex)
    cp=[creation(i) for i in range(q)];cm=[annihilation(i) for i in range(q)]
    z=[{(2*i,2*i+1):1j} for i in range(q)]
    neutral=list(z);pairs=[];cubic=[]
    # Neutral order: diagonal, then real and imaginary off-diagonals.
    for i in range(q):
        for j in range(i+1,q):
            x=clifford_product(cp[i],cm[j]);y=clifford_product(cp[j],cm[i])
            neutral.extend([combine((1/math.sqrt(2),x),(1/math.sqrt(2),y)),
                            combine((1j/math.sqrt(2),x),(-1j/math.sqrt(2),y))])
            pairs.append(clifford_product(cp[i],cp[j]))
            for k in range(q):
                if k==j:poly=clifford_product(cp[i],z[j])
                elif k==i:poly=combine((-1,clifford_product(cp[j],z[i])))
                else:poly=clifford_product(clifford_product(cp[i],cp[j]),cm[k])
                cubic.append(poly)
    return dict(labels=labels,index=indices,u1=matrix(cp,1),u0=matrix(neutral,2),
                u2=matrix(pairs,2),u3=matrix(cubic,3))

def majorana_pauli(indices,n):
    x=z=0;k=0
    for j in indices:
        p=j//2;xx=1<<p;zz=(1<<p)-1
        if j%2:zz|=1<<p
        k=(k+(j%2)+2*((z&xx).bit_count()%2))%4
        x^=xx;z^=zz
    k=(k+(len(indices)*(len(indices)-1)//2))%4
    k=(k-(x&z).bit_count())%4
    label=''.join('Y' if ((x>>p)&1 and (z>>p)&1) else 'X' if (x>>p)&1 else 'Z' if (z>>p)&1 else 'I' for p in range(n))
    return label,(1j)**k

def pauli_majorana(label):
    suffix=0;idx=[]
    for p in range(len(label)-1,-1,-1):
        x=int(label[p] in 'XY');z=int(label[p] in 'YZ')
        odd=z^suffix;even=x^odd
        if even:idx.append(2*p)
        if odd:idx.append(2*p+1)
        suffix^=x
    idx=tuple(sorted(idx));back,phase=majorana_pauli(idx,len(label))
    if back!=label:raise AssertionError('JW inverse failed')
    return idx,phase

def channels(case,cut):
    n=int(case['n_qubits']);ba=local_basis(cut);bb=local_basis(n-cut)
    dims=lambda b,d:len(b['labels'][d])
    blocks={g:np.zeros((dims(ba,g[0]),dims(bb,g[1])),complex) for g in [(1,1),(1,3),(3,1),(2,2)]}
    local2=local4=0.;unallowed=0.;total=0.
    for label,c in zip(case['pauli_labels'],case['coefficients']):
        s,phase=pauli_majorana(str(label));coef=c/phase
        total+=abs(coef)**2
        if len(s) not in [2,4]:unallowed+=abs(coef)**2;continue
        sa=tuple(j for j in s if j<2*cut);sb=tuple(j-2*cut for j in s if j>=2*cut)
        if not sa or not sb:
            if len(s)==2:local2+=abs(coef)**2
            else:local4+=abs(coef)**2
            continue
        g=(len(sa),len(sb))
        if g not in blocks:unallowed+=abs(coef)**2;continue
        # Global Hermitian Majorana -> twisted-left ordinary tensor basis.
        twist=(1j)**(g[0]*g[1]-1) if g[0]%2 else (1j)**(g[0]*g[1])
        blocks[g][ba['index'][g[0]][sa],bb['index'][g[1]][sb]]+=coef*twist
    def transform(d,l,r):
        return np.asarray((l.conj().T@d)@r.conj())
    # Right charge-minus is the conjugate of right charge-plus.
    a=transform(blocks[1,1],ba['u1'],bb['u1'].conj())
    b=transform(blocks[1,3],ba['u1'],bb['u3'].conj())
    c=transform(blocks[3,1],ba['u3'],bb['u1'].conj())
    k=transform(blocks[2,2],ba['u0'],bb['u0'])
    d2=transform(blocks[2,2],ba['u2'],bb['u2'].conj())
    def recon(o,l,r):return np.asarray((l@o)@r.T)
    leakage2=leakage4=0.
    for g,o,l,r in [((1,1),a,ba['u1'],bb['u1'].conj()),
                     ((1,3),b,ba['u1'],bb['u3'].conj()),
                     ((3,1),c,ba['u3'],bb['u1'].conj())]:
        pred=recon(o,l,r);pred=pred+pred.conj()
        err=float(np.linalg.norm(blocks[g]-pred)**2)
        if g==(1,1):leakage2+=err
        else:leakage4+=err
    pred=recon(k,ba['u0'],bb['u0'])
    pair=recon(d2,ba['u2'],bb['u2'].conj());pred+=pair+pair.conj()
    leakage4+=float(np.linalg.norm(blocks[2,2]-pred)**2)
    metrics=dict(coefficient_norm=math.sqrt(total),local_quadratic_norm=math.sqrt(local2),
        local_quartic_norm=math.sqrt(local4),unallowed_degree_norm=math.sqrt(unallowed),
        charge_projection_error=math.sqrt(leakage2+leakage4),
        quadratic_charge_leakage=math.sqrt(leakage2),quartic_charge_leakage=math.sqrt(leakage4),
        hopping_norm=float(math.sqrt(2)*np.linalg.norm(a)),transfer13_norm=float(math.sqrt(2)*np.linalg.norm(b)),
        transfer31_norm=float(math.sqrt(2)*np.linalg.norm(c)),neutral22_norm=float(np.linalg.norm(k)),
        pair22_norm=float(math.sqrt(2)*np.linalg.norm(d2)),
        imaginary_neutral_norm=float(np.linalg.norm(k.imag)))
    return dict(a=a,b=b,c=c,k=k,d2=d2,ba=ba,bb=bb,blocks=blocks,metrics=metrics)

def svd(x):
    if min(x.shape)==0:return np.zeros((x.shape[0],0),complex),np.zeros(0),np.zeros((0,x.shape[1]),complex)
    return linalg.svd(x,full_matrices=False,check_finite=False,lapack_driver='gesdd')
def values(x):return svd(x)[1]
def b4(q):return math.sqrt(math.comb(q,2)) if q>=6 else math.sqrt(15)
def c22(a,b):
    return min(math.sqrt(math.comb(2*a,2)*b),math.sqrt(math.comb(2*b,2)*a),b4(a+b))
def trace_rotation(q):
    # First q neutral coordinates are diagonal orbital matrices.
    direction=np.zeros((q*q,1));direction[:q,0]=1/math.sqrt(q)
    return linalg.qr(direction,mode='full')[0]
def sector_constants(a,b,N):
    vals=[];tt=[]
    for r in range(max(0,N-b),min(a,N)+1):
        t=N-r
        vals.append(math.sqrt((a+4*r*(a-r))*(b+4*t*(b-t))))
        tt.append(4*math.sqrt(r*(a-r)*(a+1)/a*t*(b-t)*(b+1)/b))
    return max(vals,default=0.),max(tt,default=0.)
def degree_operator(d,ba,bb,n,cut):
    from openfermion import QubitOperator,get_sparse_operator
    op=QubitOperator()
    for i,j in zip(*np.nonzero(d)):
        s=ba['labels'][2][i]+tuple(t+2*cut for t in bb['labels'][2][j])
        label,phase=majorana_pauli(s,n)
        key=tuple((p,ch) for p,ch in enumerate(label) if ch!='I')
        op.terms[key]=op.terms.get(key,0)+d[i,j]*phase
    return get_sparse_operator(op,n_qubits=n)
