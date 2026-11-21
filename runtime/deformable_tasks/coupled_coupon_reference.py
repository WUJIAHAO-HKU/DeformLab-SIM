"""Independent small-strain implicit FEM + dynamic platen + unilateral contact.

Diagnostic CPU reference, not the native finite-strain backend. No prescribed
soft nodes, velocity smoothing, penetration clipping, or trajectory overwrites.
"""
from collections import OrderedDict
import numpy as np
from scipy.sparse import coo_matrix,diags,block_diag
from scipy.sparse.linalg import splu
from scipy.linalg import cho_factor,cho_solve


def assemble_linear_tets(rest,tets,young,poisson,density):
    x=np.asarray(rest,float);tet=np.asarray(tets,int)
    if not (young>0 and density>0 and 0<=poisson<.5):raise ValueError('Invalid material')
    pts=x[tet];edge=(pts[:,1:]-pts[:,:1]).transpose(0,2,1)
    volume=abs(np.linalg.det(edge))/6
    if not np.all(volume>1e-18):raise ValueError('Degenerate mesh')
    gradient=np.zeros((len(tet),4,3));gradient[:,1:]=np.linalg.inv(edge)
    gradient[:,0]=-gradient[:,1:].sum(1)
    mu=young/(2*(1+poisson));lam=young*poisson/((1+poisson)*(1-2*poisson))
    C=np.zeros((6,6));C[:3,:3]=lam;C[np.arange(3),np.arange(3)]+=2*mu;C[3:,3:]=mu*np.eye(3)
    B=np.zeros((len(tet),6,12))
    for i in range(4):
        gx,gy,gz=gradient[:,i].T;j=3*i
        B[:,0,j]=gx;B[:,1,j+1]=gy;B[:,2,j+2]=gz
        B[:,3,j]=gy;B[:,3,j+1]=gx;B[:,4,j+1]=gz;B[:,4,j+2]=gy;B[:,5,j]=gz;B[:,5,j+2]=gx
    Ke=np.einsum('tai,ab,tbj,t->tij',B,C,B,volume)
    dofs=(3*tet[:,:,None]+np.arange(3)).reshape(-1,12)
    rows=np.broadcast_to(dofs[:,:,None],Ke.shape).ravel();cols=np.broadcast_to(dofs[:,None,:],Ke.shape).ravel()
    K=coo_matrix((Ke.ravel(),(rows,cols)),shape=(x.size,x.size)).tocsc();K=(K+K.T)*.5
    mass=np.zeros(len(x))
    for i in range(4):np.add.at(mass,tet[:,i],density*volume/4)
    if not np.all(mass>0):raise ValueError('Unreferenced node')
    return K,mass,volume


class ContactQP:
    """Warm active-set solve of a positive-definite contact Schur complement."""
    def __init__(self,S):
        self.S=np.asarray(S,float);self.diagonal=np.diag(self.S);self.active=np.zeros(len(S),bool);self.cache=OrderedDict()
        if not np.all(self.diagonal>0):raise ValueError('Nonpositive contact compliance')
        cho_factor(self.S)  # Independent admission: contact rows must be independent.

    def solve(self,gfree,tolerance=1e-11):
        active=self.active.copy();seen=set()
        for iteration in range(1,513):
            key=active.tobytes()
            if key in seen:raise RuntimeError('Contact active-set cycle; no silent fallback')
            seen.add(key);lam=np.zeros(len(gfree));ids=np.flatnonzero(active)
            if len(ids):
                if key not in self.cache:
                    self.cache[key]=cho_factor(self.S[np.ix_(ids,ids)],check_finite=False)
                    if len(self.cache)>64:self.cache.popitem(last=False)
                lam[ids]=cho_solve(self.cache[key],-gfree[ids],check_finite=False)
            gap=gfree+self.S@lam
            natural=lam-np.maximum(0.,lam-gap/self.diagonal)
            residual=float(np.max(abs(natural)*self.diagonal,initial=0))
            if residual<=tolerance and lam.min(initial=0)>=-1e-9:
                self.active=active
                return lam,gap,dict(iterations=iteration,natural_residual_m=residual,active=int(active.sum()))
            # Principal active-set update: negative reactions release; violated
            # inactive constraints activate. No change to geometry tolerance.
            active=(lam-gap/self.diagonal)>0
        raise RuntimeError('Contact QP did not converge')


class CoupledCoupon:
    def __init__(self,rest,tets,young,poisson,density,hz,linear_damping=.1,kelvin_voigt_s=0.,
                 table_z=.12,plate_bottom=.202,tool_mass=.2,drive_stiffness=1e6,drive_damping=4000.):
        parameters=[young,poisson,density,hz,linear_damping,kelvin_voigt_s,table_z,plate_bottom,tool_mass,drive_stiffness,drive_damping]
        if not np.isfinite(parameters).all() or hz<=0 or linear_damping<0 or kelvin_voigt_s<0 or tool_mass<=0 or drive_stiffness<0 or drive_damping<0:
            raise ValueError('Invalid integration parameters')
        self.rest=np.asarray(rest,float);self.tets=np.asarray(tets,int);self.n=len(rest);self.dt=1/float(hz)
        if self.rest.ndim!=2 or self.rest.shape[1]!=3 or not np.isfinite(self.rest).all():raise ValueError('Invalid rest coordinates')
        if self.tets.ndim!=2 or self.tets.shape[1]!=4 or self.tets.min()<0 or self.tets.max()>=self.n:raise ValueError('Invalid tetrahedra')
        self.Ksoft,self.mass,self.vol=assemble_linear_tets(rest,tets,young,poisson,density)
        self.table_z=table_z;self.plate_bottom=plate_bottom;self.k=drive_stiffness;self.c=drive_damping
        self.M=np.r_[np.repeat(self.mass,3),tool_mass];self.K=block_diag((self.Ksoft,np.zeros((1,1))),format='csc')
        self.Dsoft=linear_damping*diags(np.repeat(self.mass,3))+kelvin_voigt_s*self.Ksoft
        self.D=block_diag((self.Dsoft,np.array([[drive_damping]])),format='csc')
        drive_diag=np.zeros(len(self.M));drive_diag[-1]=drive_stiffness;self.Ktotal=self.K+diags(drive_diag)
        h=self.dt;self.H=diags(self.M/(h*h))+self.D/h+self.Ktotal;self.factor=splu(self.H.tocsc())
        self.top=np.flatnonzero(np.isclose(self.rest[:,2],self.rest[:,2].max(),atol=1e-8,rtol=0))
        self.bottom=np.flatnonzero(np.isclose(self.rest[:,2],self.rest[:,2].min(),atol=1e-8,rtol=0))
        # Planar reference: only the original extreme faces can touch. Every
        # vertex and positive volume is checked after each step to enforce scope.
        nt=len(self.top);nb=len(self.bottom);B=np.zeros((nt+nb,len(self.M)))
        B[np.arange(nt),3*self.top+2]=-1.;B[:nt,-1]=1.;B[nt+np.arange(nb),3*self.bottom+2]=1.
        self.B=B;self.g0=np.r_[plate_bottom-self.rest[self.top,2],self.rest[self.bottom,2]-table_z]
        if self.g0.min()<-1e-10:raise ValueError('Reference requires initially feasible geometry')
        self.X=self.factor.solve(B.T);self.S=B@self.X;self.qp=ContactQP((self.S+self.S.T)*.5)
        self.u=np.zeros(len(self.M));self.v=np.zeros(len(self.M));self.target=0.;self.step_index=0

    def step(self,target):
        if not np.isfinite(target):raise ValueError('Nonfinite drive target')
        h=self.dt;old=self.u.copy();v0=self.v.copy();target0=self.target
        # Solve for the increment directly. Forming M*u_old/h**2 then
        # subtracting u_old after the solve loses velocity/force accuracy at
        # small h, precisely where adaptive impact refinement needs it most.
        rhs=self.M*v0/h-self.Ktotal@old;rhs[-1]+=self.k*target
        free_increment=self.factor.solve(rhs)
        # A fixed length tolerance becomes a velocity error proportional to 1/h.
        # Keep the admission tied to normal velocity resolution when h shrinks.
        self.position_contact_tolerance=max(5e-18,min(1e-11,h*1e-9))
        lam,gap,qp=self.qp.solve(self.g0+self.B@old+self.B@free_increment,tolerance=self.position_contact_tolerance)
        du=free_increment+self.X@lam;u=old+du;v=du/h;dv=v-v0
        drive=self.k*(target-u[-1])-self.c*v[-1]
        elastic=self.K@u;damping=np.r_[self.Dsoft@v[:-1],0.]
        applied=np.zeros(len(u));applied[-1]=drive
        force_residual=self.M*dv/h+elastic+damping-applied-self.B.T@lam
        kinetic_delta=.5*np.sum(self.M*(v*v-v0*v0));elastic_delta=.5*(u@(self.K@u)-old@(self.K@old))
        drive_work=drive*du[-1];contact_work=float(lam@(self.B@du));dissipation=h*float(v[:-1]@(self.Dsoft@v[:-1]))
        numerical_loss=.5*np.sum(self.M*dv*dv)+.5*float(du@(self.K@du))
        residual=kinetic_delta+elastic_delta-drive_work-contact_work+dissipation+numerical_loss
        target_work=.5*self.k*((old[-1]-target)**2-(old[-1]-target0)**2)
        spring_delta=.5*self.k*((u[-1]-target)**2-(old[-1]-target0)**2)
        full_loss=numerical_loss+.5*self.k*du[-1]**2
        drive_dissipation=h*self.c*v[-1]**2
        full_residual=kinetic_delta+elastic_delta+spring_delta-target_work-contact_work+dissipation+drive_dissipation+full_loss
        x=self.rest+u[:-1].reshape(-1,3);xt=x[self.tets]
        J=np.linalg.det((xt[:,1:]-xt[:,:1]).transpose(0,2,1))/(6*self.vol)
        # Tet orientation is allowed to be either sign in the original mesh.
        rt=self.rest[self.tets];J*=np.sign(np.linalg.det((rt[:,1:]-rt[:,:1]).transpose(0,2,1)))
        penetration=max(0.,float(self.table_z-x[:,2].min()),float(x[:,2].max()-(self.plate_bottom+u[-1])))
        if not np.isfinite(u).all() or J.min()<=0 or penetration>1e-9:raise RuntimeError('Reference left admissible planar regime')
        if np.max(abs(force_residual))>1e-6 or abs(residual)>1e-9 or abs(full_residual)>1e-9 or contact_work>1e-9:
            raise RuntimeError(f'Coupled force/work consistency failed: h={h:.9g}, force={abs(force_residual).max():.9g}, work={residual:.9g}, full_work={full_residual:.9g}, contact_work={contact_work:.9g}')
        self.u=u;self.v=v;self.target=target;self.step_index+=1
        nodal_v=v[:-1].reshape(-1,3);com_v=(self.mass[:,None]*nodal_v).sum(0)/self.mass.sum()
        internal_force_sum=(self.Ksoft@u[:-1]).reshape(-1,3).sum(0)
        self.last_contact_force=lam.copy()
        self.last_integrated_velocity=v.copy()
        self.last_motion_velocity=v.copy()
        self.last_position_increment=u-old
        return dict(step=self.step_index,time_s=self.step_index*h,tool_q_m=float(u[-1]),tool_v_m_s=float(v[-1]),
            drive_force_N=float(drive),reaction_N=float(lam[:len(self.top)].sum()),bottom_reaction_N=float(lam[len(self.top):].sum()),
            strain=float(1-(x[self.top,2].mean()-x[self.bottom,2].mean())/(self.rest[:,2].max()-self.rest[:,2].min())),
            max_nodal_speed_m_s=float(np.linalg.norm(v[:-1].reshape(-1,3),axis=1).max()),max_penetration_m=penetration,min_J=float(J.min()),
            com_vx_m_s=float(com_v[0]),com_vy_m_s=float(com_v[1]),com_vz_m_s=float(com_v[2]),
            max_translation_removed_speed_m_s=float(np.linalg.norm(nodal_v-com_v,axis=1).max()),
            internal_force_sum_N=float(np.linalg.norm(internal_force_sum)),
            force_residual_N=float(abs(force_residual).max()),kinetic_change_J=float(kinetic_delta),elastic_change_J=float(elastic_delta),
            drive_work_J=float(drive_work),contact_work_J=contact_work,material_drag_dissipation_J=dissipation,
            integration_loss_J=float(numerical_loss),mechanical_work_residual_J=float(residual),
            drive_target_work_J=float(target_work),drive_spring_change_J=float(spring_delta),drive_dissipation_J=float(drive_dissipation),
            total_integration_loss_J=float(full_loss),full_work_residual_J=float(full_residual),**qp)


class ImpactCoupledCoupon(CoupledCoupon):
    """Position-feasible implicit step followed by an explicit inelastic impact.

    The second stage changes momentum at fixed position. Its impulse and work
    are recorded separately; post-impact velocity is NOT displacement / dt.
    This is endpoint time stepping, not exact continuous collision detection.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.impact_schur=(self.B/self.M)@self.B.T
        self.impact_cache=OrderedDict()

    def integrated_step(self,target):
        return super().step(target)

    def step(self,target):
        initial_velocity=self.v.copy()
        row=self.integrated_step(target);pre=self.v.copy();h=self.dt
        gap=self.g0+self.B@self.u
        closed=gap<=max(5e-17,2*self.position_contact_tolerance);ids=np.flatnonzero(closed);impulse=np.zeros(len(gap));iterations=0
        if len(ids):
            key=closed.tobytes()
            if key not in self.impact_cache:
                self.impact_cache[key]=ContactQP(self.impact_schur[np.ix_(ids,ids)])
                if len(self.impact_cache)>64:self.impact_cache.popitem(last=False)
            impulse[ids],_,info=self.impact_cache[key].solve((self.B@pre)[ids],tolerance=1e-11)
            iterations=info['iterations']
        delta=self.B.T@impulse/self.M;post=pre+delta
        work=float(impulse@(self.B@(pre+.5*delta)))
        kinetic_jump=.5*float(np.sum(self.M*(post*post-pre*pre)))
        impulse_residual=self.M*(post-pre)-self.B.T@impulse
        residual=kinetic_jump-work
        inward=max(0.,float(-(self.B@post)[ids].min())) if len(ids) else 0.
        if work>1e-10 or abs(residual)>1e-10 or abs(impulse_residual).max()>1e-10 or inward>1e-9:
            raise RuntimeError('Inelastic impact feasibility/work admission failed')
        self.v=post
        row['reaction_N']+=float(impulse[:len(self.top)].sum()/h)
        row['bottom_reaction_N']+=float(impulse[len(self.top):].sum()/h)
        row['kinetic_change_J']+=kinetic_jump
        row['contact_work_J']+=work
        row['mechanical_work_residual_J']+=residual;row['full_work_residual_J']+=residual
        nodal=post[:-1].reshape(-1,3);com=(self.mass[:,None]*nodal).sum(0)/self.mass.sum()
        row.update(tool_v_m_s=float(post[-1]),max_nodal_speed_m_s=float(np.linalg.norm(nodal,axis=1).max()),
            com_vx_m_s=float(com[0]),com_vy_m_s=float(com[1]),com_vz_m_s=float(com[2]),
            max_translation_removed_speed_m_s=float(np.linalg.norm(nodal-com,axis=1).max()),
            impact_work_J=work,impact_kinetic_jump_J=kinetic_jump,impact_work_residual_J=residual,
            impact_impulse_residual_Ns=float(abs(impulse_residual).max()),impact_iterations=iterations,
            impact_top_impulse_Ns=float(impulse[:len(self.top)].sum()),impact_bottom_impulse_Ns=float(impulse[len(self.top):].sum()),
            closed_inward_speed_m_s=inward,
            position_integrated_velocity_residual_m=float(abs(self.last_position_increment-h*self.last_motion_velocity).max()),
            peak_preimpact_nodal_speed_m_s=float(np.linalg.norm(pre[:-1].reshape(-1,3),axis=1).max()))
        # Linear-FEM angular momentum is taken about fixed rest coordinates.
        # Full finite-rotation angular conservation is outside this reference.
        contact_impulse=self.B.T@(h*self.last_contact_force+impulse)
        soft_drag_impulse=-h*(self.Dsoft@self.last_motion_velocity[:-1]).reshape(-1,3)
        nodal_change=(self.mass[:,None]*(post[:-1]-initial_velocity[:-1]).reshape(-1,3))
        contact_nodes=contact_impulse[:-1].reshape(-1,3)
        angular_change=np.cross(self.rest,nodal_change).sum(0)
        angular_external=np.cross(self.rest,contact_nodes+soft_drag_impulse).sum(0)
        row['linearized_angular_impulse_residual_Nms']=float(abs(angular_change-angular_external).max())
        for axis,index in zip('xyz',range(3)):
            row['contact_angular_impulse_'+axis+'_Nms']=float(np.cross(self.rest,contact_nodes).sum(0)[index])
        # Preserve integrated (pre-impact) drag for independent global audit.
        row['soft_drag_impulse_z_Ns']=float(soft_drag_impulse[:,2].sum())
        self.last_impact_impulse=impulse
        return row
