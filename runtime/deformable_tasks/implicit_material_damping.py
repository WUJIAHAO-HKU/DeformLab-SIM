"""Opt-in whole-body implicit velocity damping with physical nodal masses.

Elastic stepping uses zero native elasticity damping. This separate dissipative
operator implements the requested coefficient; it does not advance positions.
Dense CUDA factorization is a diagnostic reference, not a performance claim.
"""
import json,sys
import numpy as np

class DampingSystem:
    def __init__(self,rest,tets,mass,young,poisson,gamma):
        import torch
        self.torch=torch;self.tets=tets;self.mass=mass;self.n=len(mass);self.gamma=gamma
        self.mu=young/(2*(1+poisson));self.poisson=poisson
        rt=rest[tets];Dm=(rt[:,1:]-rt[:,:1]).transpose(1,2)
        self.Q=torch.linalg.inv(Dm);self.vol=torch.abs(torch.linalg.det(Dm))/6
        dofs=(3*tets[:,:,None]+torch.arange(3,device=rest.device)[None,None,:]).reshape(-1,12)
        self.rows=dofs[:,:,None].expand(-1,12,12).reshape(-1)
        self.cols=dofs[:,None,:].expand(-1,12,12).reshape(-1)
        self.inv_sqrt_mass=mass.rsqrt().repeat_interleave(3)
        self.identity=torch.eye(3*self.n,device=rest.device,dtype=rest.dtype)

    def solve(self,positions,velocities,dt,integrator='backward-euler'):
        if integrator not in ('backward-euler','exponential'):raise ValueError('Unknown damping integrator')
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite damping step required')
        torch=self.torch;x=positions[self.tets];F=(x[:,1:]-x[:,:1]).transpose(1,2)@self.Q
        J=torch.linalg.det(F)
        if not bool((J>0).all()):raise ValueError('Implicit damping requires positive tetrahedra')
        u,s,vh=torch.linalg.svd(F);R=u@vh
        strain=F-R;c=torch.sqrt((strain*strain).sum((1,2))+1e-18)
        def gradient(T):
            corners=(T@self.Q.transpose(1,2)).transpose(1,2)
            return torch.cat([-corners.sum(1,keepdim=True),corners],dim=1).reshape(-1,12)
        g0=gradient(strain/c[:,None,None]);g1=gradient(J[:,None,None]*torch.linalg.inv(F).transpose(1,2))
        block=g0[:,:,None]*g0[:,None,:]
        if self.poisson>0:block=block+g1[:,:,None]*g1[:,None,:]
        block=block*(2*self.mu*self.gamma*self.vol[:,None,None])
        D=torch.zeros_like(self.identity)
        D.index_put_((self.rows,self.cols),block.reshape(-1),accumulate=True)
        w=self.inv_sqrt_mass
        S=w[:,None]*D*w[None,:];S=(S+S.T)*.5
        rhs=velocities.reshape(-1)/w
        if integrator=='backward-euler':
            A=self.identity+dt*S;factor=torch.linalg.cholesky(A)
            solved=torch.cholesky_solve(rhs[:,None],factor)[:,0]
            residual=torch.linalg.vector_norm(A@solved-rhs)/(torch.linalg.vector_norm(rhs)+1e-30)
            physical=dt*(solved@(S@solved));numerical=.5*((solved-rhs)**2).sum()
            spectrum={}
        else:
            eigen,U=torch.linalg.eigh(S)
            scale=max(float(abs(eigen).max()),1.)
            if float(eigen.min()) < -1e-10*scale:raise RuntimeError('Indefinite damping operator')
            # Keep roundoff eigenvalues, including tiny negatives, observable.
            coefficients=U.T@rhs
            solved=U@(torch.exp(-dt*eigen)*coefficients)
            residual=torch.linalg.vector_norm(S@U-U*eigen[None,:])/(torch.linalg.vector_norm(S)+1e-30)
            # Independent integral of v(t)^T D v(t) in the spectral basis.
            physical=.5*(coefficients**2*(-torch.expm1(-2*dt*eigen))).sum()
            numerical=torch.zeros((),device=rhs.device,dtype=rhs.dtype)
            spectrum=dict(minimum_eigenvalue_s_inv=float(eigen.min()),maximum_eigenvalue_s_inv=float(eigen.max()))
        new=(solved*w).reshape(-1,3)
        dK=.5*(self.mass[:,None]*(new*new-velocities*velocities)).sum()
        self.last_ledger=dict(integrator=integrator,kinetic_change_J=float(dK),physical_dissipation_J=float(physical),
            integration_dissipation_J=float(numerical),work_residual_J=float(dK+physical+numerical),
            operator_residual=float(residual),residual_kind='linear_system' if integrator=='backward-euler' else 'spectral_decomposition',**spectrum)
        return new,float(residual)

class ImplicitMaterialDamping:
    def __init__(self,root,runtime,policy):
        self.active=policy is not None;self.steps=0;self.max_residual=0.;self.max_impulse=0.;self.max_angular=0.;self.work=0.;self.max_energy_increase=0.
        self.ledger=[];self.max_work_residual=0.;self.physical_loss=0.;self.numerical_loss=0.;self.publication_work=0.
        if not self.active:return
        import torch,warp as wp
        from omni.physx import get_physx_interface
        sys.path.insert(0,str(root/'work/native-readback'))
        import physx_deformable_readback as readback
        if len(runtime.bodies)!=1:raise ValueError('Damping reference supports one volumetric body')
        self.runtime=runtime;self.policy=policy;self.integrator=policy.get('integrator','backward-euler');b=runtime.bodies[0]
        handle=readback.simulation_buffer(get_physx_interface(),b['path'])
        self.wp_p=wp.array(ptr=handle['device_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0')
        self.wp_v=wp.array(ptr=handle['velocity_pointer'],shape=(handle['node_count'],),dtype=wp.vec4,device='cuda:0')
        self.p=wp.to_torch(self.wp_p);self.v=wp.to_torch(self.wp_v)
        self.mass=1/self.v[:,3].double();self.gamma=policy['gamma'];self.torch=torch
        rest=torch.as_tensor(b['rest'],dtype=torch.float64,device='cuda')
        tets=torch.as_tensor(b['sim_elements'].astype(np.int64),device='cuda')
        self.system=DampingSystem(rest,tets,self.mass,runtime.config['material']['young_pa'],runtime.config['material']['poisson'],self.gamma)

    def apply(self):
        if not self.active:return
        torch=self.torch;self.steps+=1
        x=self.p[:,:3].double();v=self.v[:,:3].double()
        new,residual=self.system.solve(x,v,1/self.runtime.config['solver']['hz'],self.integrator)
        published=new.float().double();dv=published-v;m=self.mass[:,None]
        impulse=torch.linalg.vector_norm((m*dv).sum(0))
        center=(m*x).sum(0)/m.sum();angular=torch.linalg.vector_norm(torch.cross(x-center,m*dv,dim=1).sum(0))
        dK=float(.5*(m*(published*published-v*v)).sum())
        publication=float(.5*(m*(published*published-new*new)).sum())
        entry=dict(self.system.last_ledger,step=self.steps,published_kinetic_change_J=dK,
            publication_energy_change_J=publication,linear_impulse_Ns=float(impulse),angular_impulse_Nms=float(angular))
        full_residual=dK+entry['physical_dissipation_J']+entry['integration_dissipation_J']-publication
        entry['published_work_residual_J']=full_residual;self.ledger.append(entry)
        self.max_work_residual=max(self.max_work_residual,abs(full_residual))
        self.physical_loss+=entry['physical_dissipation_J'];self.numerical_loss+=entry['integration_dissipation_J'];self.publication_work+=publication
        self.max_residual=max(self.max_residual,residual);self.max_impulse=max(self.max_impulse,float(impulse));self.max_angular=max(self.max_angular,float(angular))
        self.work+=dK;self.max_energy_increase=max(self.max_energy_increase,dK)
        if residual>1e-8 or float(impulse)>1e-7 or float(angular)>1e-8 or dK>1e-8 or abs(full_residual)>1e-8 or entry['physical_dissipation_J'] < -1e-8:
            raise RuntimeError('Implicit material damping failed conservation/dissipation checks')
        self.v[:,:3].copy_(published.float());torch.cuda.synchronize()

    def report(self):
        if not self.active:return {}
        return dict(steps=self.steps,requested_gamma=self.gamma,native_gamma=0.,integrator=self.integrator,maximum_linear_impulse_Ns=self.max_impulse,
            maximum_angular_impulse_Nms=self.max_angular,maximum_operator_residual=self.max_residual,
            residual_kind='linear_system' if self.integrator=='backward-euler' else 'spectral_decomposition',
            maximum_work_residual_J=self.max_work_residual,physical_dissipation_J=self.physical_loss,
            integration_dissipation_J=self.numerical_loss,publication_energy_change_J=self.publication_work,step_ledger=self.ledger,
            **({'maximum_linear_system_residual':self.max_residual} if self.integrator=='backward-euler' else {}),
            kinetic_work_J=self.work,maximum_kinetic_increase_J=self.max_energy_increase,
            accepted=self.steps==self.runtime.steps,
            scope='Opt-in whole-frame directional material velocity damping. Native elasticity damping zero; positions fixed during this substep. Dense CUDA diagnostic, no performance or whole-step coupling claim.')
