"""Linear-elastic material-response envelope used by the refresh schedule.
Requires Python 3.10+, NumPy 1.24+ and SciPy 1.10+.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
from collections import Counter

def assemble(rest,tet,young,nu):
    x=np.asarray(rest,dtype=float);tet=np.asarray(tet,dtype=int)
    edges=x[tet[:,1:]]-x[tet[:,0,None]]
    vol=np.abs(np.linalg.det(edges))/6
    assert np.all(vol>0)
    grad=np.zeros((len(tet),4,3));grad[:,1:]=np.swapaxes(np.linalg.inv(edges),1,2);grad[:,0]=-grad[:,1:].sum(1)
    assert np.allclose(np.einsum('tni,tnj->tij',x[tet],grad),np.eye(3),atol=2e-12)
    b=np.zeros((len(tet),6,12))
    for n in range(4):
        gx,gy,gz=grad[:,n].T;k=n*3
        b[:,0,k]=gx;b[:,1,k+1]=gy;b[:,2,k+2]=gz
        b[:,3,k]=gy;b[:,3,k+1]=gx
        b[:,4,k+1]=gz;b[:,4,k+2]=gy
        b[:,5,k]=gz;b[:,5,k+2]=gx
    mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    c=np.zeros((6,6));c[:3,:3]=lam;np.fill_diagonal(c[:3,:3],lam+2*mu);c[3:,3:]=np.eye(3)*mu
    ke=np.einsum('tka,kl,tlb,t->tab',b,c,b,vol,optimize=True)
    ids=(3*tet[:,:,None]+np.arange(3)).reshape(-1,12)
    rr=np.broadcast_to(ids[:,:,None],ke.shape).ravel();cc=np.broadcast_to(ids[:,None,:],ke.shape).ravel()
    k=coo_matrix((ke.ravel(),(rr,cc)),shape=(3*len(x),3*len(x))).tocsr();k.sum_duplicates()
    faces=Counter(tuple(sorted(int(t[i])for i in f))for t in tet for f in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
    assert max(faces.values())==2
    boundary=np.array(sorted({v for f,n in faces.items()if n==1 for v in f}),dtype=int)
    trans=np.tile(np.eye(3),(len(x),1));scale=max(float(np.max(abs(k.diagonal()))),1e-300)
    rigid_residual=float(np.max(abs(k@trans))/scale)
    symmetry=k-k.T;symmetry_residual=float(np.max(abs(symmetry.data))/scale)if symmetry.nnz else 0.
    assert rigid_residual<1e-11 and symmetry_residual<1e-12
    return k,boundary,dict(rigid_translation_residual=rigid_residual,symmetry_residual=symmetry_residual)

def response(k,boundary,mass,h):
    a=(k+diags(np.repeat(mass/h**2,3))).tocsc();lu=splu(a)
    n=len(mass);gain=np.zeros(n);stiffness=np.zeros(n);max_residual=0.;min_c=float('inf')
    weights=mass/mass.sum()
    for start in range(0,len(boundary),24):
        nodes=boundary[start:start+24];rhs=np.zeros((3*n,len(nodes)))
        rhs[3*nodes+2,np.arange(len(nodes))]=1
        z=lu.solve(rhs);c=z[3*nodes+2,np.arange(len(nodes))]
        assert np.all(c>0);min_c=min(min_c,float(c.min()))
        max_residual=max(max_residual,float(np.max(abs(a@z-rhs))))
        field=z.reshape(n,3,-1);center=np.einsum('n,ncb->cb',weights,field)
        horizontal=field[:,:2,:]-center[None,:2,:]
        gain[nodes]=np.linalg.norm(horizontal,axis=1).max(0)/c
        stiffness[nodes]=1/c
    assert max_residual<1e-9
    # Exact checks of the stated single-contact linear reference, with 1um correction.
    j=int(boundary[len(boundary)//2]);rhs=np.zeros(3*n);rhs[3*j+2]=1
    z=lu.solve(rhs);g=1e-6;u=z*g/z[3*j+2];f=g/z[3*j+2]
    kkt=float(np.max(abs(a@u-rhs*f))/max(abs(f),1e-20))
    work=float(u@(a@u));contact_work=float(f*g)
    assert kkt<1e-9 and np.isclose(work,contact_work,rtol=1e-10,atol=1e-20)
    return gain,stiffness,dict(linear_solve_max_residual=max_residual,single_contact_kkt_residual=kkt,
        single_contact_work_relative_error=abs(work-contact_work)/contact_work,
        minimum_normal_compliance_m_per_n=min_c,gain_min=float(gain[boundary].min()),gain_max=float(gain.max()),gain_median=float(np.median(gain[boundary])))

