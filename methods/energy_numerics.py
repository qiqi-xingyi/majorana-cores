"""Physical queries and conservative floating-point enclosures for energy error bounds."""
from __future__ import annotations
from fractions import Fraction
import math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, eigsh

EPS = np.finfo(float).eps


def gamma(count):
    value = int(count)*EPS
    if value >= 0.5:
        raise ValueError("Floating-point enclosure operation count is too large")
    return value/(1-value)


def up(value):
    return float(np.nextafter(float(value),np.inf))


def upper_norm(values):
    absolute=np.abs(np.ravel(values))
    scale=float(np.max(absolute,initial=0.0))
    if scale==0: return 0.0
    normalized=absolute/scale
    value=math.fsum(float(x)**2 for x in normalized)
    return up(scale*math.sqrt(up(value/(1-EPS)))/(1-EPS)**4)


def upper_l1(values):
    return up(math.fsum(float(abs(x)) for x in np.ravel(values)))


def residual_enclosure(c,left,right):
    nominal=c-left@right
    absolute_product=np.abs(left)@np.abs(right)
    g=gamma(max(1,left.shape[1]))
    error=np.nextafter(g/(1-g)*absolute_product + gamma(1)*np.abs(nominal)
                       + np.finfo(float).tiny,np.inf)
    absolute=np.nextafter(np.abs(nominal)+error,np.inf)
    return nominal,error,absolute


def verify_dyadic(c,left,right,absolute):
    maximum=0.0
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            value=Fraction.from_float(float(c[i,j]))
            for r in range(left.shape[1]):
                value-=Fraction.from_float(float(left[i,r]))*Fraction.from_float(float(right[r,j]))
            if abs(value)>Fraction.from_float(float(absolute[i,j])):
                raise AssertionError("Entrywise enclosure missed the exact dyadic residual")
            maximum=max(maximum,float(abs(value)))
    return dict(verified_entries=c.size,maximum_exact_residual_coefficient=maximum,passed=True)


def pauli_masks(label):
    n=len(label);x=z=ys=0
    for q,axis in enumerate(label):
        bit=1<<(n-1-q)
        if axis in "XY": x|=bit
        if axis in "ZY": z|=bit
        if axis=="Y": ys+=1
    return x,z,(1j)**ys


def pauli_sparse(labels,coefficients,n,max_nnz=20_000_000):
    """Build by computational-basis Pauli action, q=0 is the leading bit."""
    dimension=2**n
    basis=np.arange(dimension,dtype=np.int64)
    parity=np.fromiter((i.bit_count()&1 for i in range(dimension)),np.int8,dimension)
    result=sparse.csr_matrix((dimension,dimension),dtype=complex)
    for start in range(0,len(labels),32):
        rows=[];columns=[];values=[]
        for label,coefficient in zip(labels[start:start+32],coefficients[start:start+32]):
            if coefficient==0: continue
            x,z,phase=pauli_masks(str(label))
            rows.append(basis^x);columns.append(basis)
            values.append(complex(coefficient)*phase*(1-2*parity[basis&z]))
        if not rows: continue
        block=sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
                                shape=(dimension,dimension)).tocsr()
        result=result+block
        result.eliminate_zeros()
        if result.nnz>max_nnz:
            raise MemoryError("Physical sparse matrix exceeded declared nnz guard")
    return result


def local_pauli_matrix(keys,coefficients):
    n=len(keys[0]);d=2**n
    value=pauli_sparse(keys,coefficients,n).toarray()
    error=up(d*gamma(2*len(keys)+2)*upper_l1(coefficients))
    return value,error


class FactorQuery:
    """Actual stored L_r/R_r Pauli factors plus a shared identity scalar."""
    def __init__(self,left,right,a_keys,b_keys,identity):
        self.identity=float(identity)
        self.da=2**len(a_keys[0]);self.db=2**len(b_keys[0])
        self.shape=(self.da*self.db,)*2;self.dtype=np.dtype(complex)
        self.pairs=[];self.local_error_fro=0.;self.norm_product_sum=0.
        self.local_hermiticity_defect=0.
        for r in range(left.shape[1]):
            a,ea=local_pauli_matrix(a_keys,left[:,r])
            b,eb=local_pauli_matrix(b_keys,right[r])
            na=upper_norm(a);nb=upper_norm(b)
            self.local_error_fro+=ea*nb+na*eb+ea*eb
            self.norm_product_sum+=na*nb
            self.local_hermiticity_defect=max(self.local_hermiticity_defect,
                float(np.linalg.norm(a-a.conj().T)),float(np.linalg.norm(b-b.conj().T)))
            self.pairs.append((a,b))
    def apply(self,vectors):
        original=np.asarray(vectors)
        is_vector=original.ndim==1
        x=original.reshape(self.da,self.db,-1)
        batch=x.shape[2]
        result=self.identity*x
        for a,b in self.pairs:
            temp=(a@x.reshape(self.da,self.db*batch)).reshape(self.da,self.db,batch)
            temp=(temp.transpose(0,2,1).reshape(self.da*batch,self.db)@b.T)
            result=result+temp.reshape(self.da,batch,self.db).transpose(0,2,1)
        shaped=result.reshape(self.shape[0],batch)
        return shaped[:,0] if is_vector else shaped
    def columns(self,indices):
        """Direct columns of the factor sum; avoids one-hot dense matmuls."""
        indices=np.asarray(indices,dtype=np.int64)
        a0=indices//self.db;b0=indices%self.db
        result=np.zeros((self.da,self.db,len(indices)),complex)
        for a,b in self.pairs:
            result+=np.einsum("ib,jb->ijb",a[:,a0],b[:,b0],optimize=False)
        result=result.reshape(self.shape[0],len(indices))
        result[indices,np.arange(len(indices))]+=self.identity
        return result
    def columns_error_fro(self,column_count):
        # Each selected entry uses complex products and a finite sum.  The
        # factor-local construction error is independent of this query path.
        arithmetic=math.sqrt(column_count)*gamma(8*len(self.pairs)+16)*(
            self.norm_product_sum+abs(self.identity))
        return up(self.local_error_fro+arithmetic)
    def linear_operator(self):
        return LinearOperator(self.shape,matvec=self.apply,matmat=self.apply,rmatvec=self.apply,dtype=complex)
    def matrix_error_fro(self,column_count):
        arithmetic=math.sqrt(column_count)*gamma(
            2*self.da+2*self.db+4*len(self.pairs)+16)*(self.norm_product_sum+abs(self.identity))
        return up(self.local_error_fro+arithmetic)
    def dense(self):
        return self.apply(np.eye(self.shape[0],dtype=complex))


def sector_projection(query,indices,batch_size=32):
    dimension=query.shape[0];count=len(indices)
    projected=np.empty((count,count),complex)
    keep=np.zeros(dimension,bool);keep[indices]=True
    leakage2=0.
    for start in range(0,count,batch_size):
        stop=min(start+batch_size,count)
        values=query.columns(indices[start:stop])
        projected[:,start:stop]=values[indices]
        leakage2+=float(np.sum(np.abs(values[~keep])**2))
    return projected,math.sqrt(leakage2),query.columns_error_fro(count)


def clean_hermitian(matrix):
    defect=float(np.linalg.norm(matrix-matrix.conj().T))
    if defect>1e-10*max(1.,float(np.linalg.norm(matrix))):
        raise ValueError("Physical query is not Hermitian within the declared roundoff tolerance")
    clean=(matrix+matrix.conj().T)/2
    return clean,defect/2


def complete_spectrum(matrix,external_error=0.):
    clean,cleanup=clean_hermitian(matrix)
    values,vectors=np.linalg.eigh(clean)
    residual=upper_norm(clean-(vectors*values)@vectors.conj().T)
    gram=upper_norm(vectors.conj().T@vectors-np.eye(len(values)))
    if not np.isfinite(gram) or gram>=0.5:
        raise ValueError("Dense eigenvectors failed the orthogonality enclosure check")
    largest=float(np.max(np.abs(values)))
    arithmetic=up(gamma(8*len(values)+16)*len(values)*max(1.,upper_norm(clean)))
    op_upper=up((1+gram)*largest+residual+arithmetic+cleanup+external_error)
    energy_uncertainty=up(residual+2*gram/(1-gram)*largest+
                          arithmetic+cleanup+external_error)
    return dict(ground=float(values[0]),ground_uncertainty=energy_uncertainty,
                op_estimate=largest,op_upper=op_upper,
                numerical_hermitian_cleanup=cleanup,
                eigendecomposition_residual_fro=residual,
                eigensolver="complete_dense_eigh_with_roundoff_enclosure")


def ground_iterative(operator,seed=20261025):
    n=operator.shape[0]
    rng=np.random.default_rng(seed)
    v0=rng.normal(size=n);v0/=np.linalg.norm(v0)
    values,vectors=eigsh(operator,k=1,which="SA",tol=1e-12,maxiter=10000,v0=v0)
    value=float(values[0]);vector=vectors[:,0]
    residual=float(np.linalg.norm(operator@vector-value*vector))
    return dict(ground=value,ground_residual=residual,
                ground_uncertainty=None,
                eigensolver="iterative_ground_estimate; Ritz residual is not a global enclosure")


def anticommuting_groups(a_keys,b_keys,absolute,max_terms=5000):
    threshold=np.finfo(float).tiny*10
    omitted=upper_l1(absolute[absolute<=threshold])
    support=np.argwhere(absolute>threshold)
    if len(support)>max_terms:
        return None,[],dict(status="not_evaluated",reason="residual support exceeds grouping guard",
                            residual_support=len(support),max_terms=max_terms)
    terms=[]
    for i,j in support:
        label=a_keys[i]+b_keys[j]
        x,z,_=pauli_masks(label)
        terms.append((float(absolute[i,j]),x,z,label))
    terms.sort(key=lambda item:(-item[0],item[3]))
    groups=[]
    for coefficient,x,z,label in terms:
        for group in groups:
            if all((((x&other[2]).bit_count()+(z&other[1]).bit_count())&1)
                   for other in group):
                group.append((coefficient,x,z,label));break
        else:
            groups.append([(coefficient,x,z,label)])
    bound=up(math.fsum(upper_norm([item[0] for item in group]) for group in groups)+omitted)
    labels=[[item[3] for item in group] for group in groups]
    return bound,labels,dict(status="evaluated",groups=len(groups),residual_support=len(support),
                            omitted_underflow_guard_l1=omitted)
