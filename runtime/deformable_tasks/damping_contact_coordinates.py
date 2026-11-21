"""Power-conjugate joint/contact coordinates for the Cartesian damping experiment.

Keeps the full measured generalized mass and Jacobian. A measured native-link
versus J*qdot mismatch is an explicit affine observation residual, not silently
absorbed into a changed mass, damping coefficient, or force measurement.
"""
import numpy as np

class ContactCoordinates:
    def __init__(self,mass,jacobian,joint_velocity,link_velocity):
        M,J,q,v=map(lambda x:np.asarray(x,dtype=float),(mass,jacobian,joint_velocity,link_velocity))
        if M.shape!=(3,3) or J.shape!=(3,3) or q.shape!=(3,) or v.shape!=(3,):raise ValueError('Three translational joint coordinates required')
        if not all(np.isfinite(x).all() for x in (M,J,q,v)):raise ValueError('Finite measured coordinates required')
        if not np.allclose(M,M.T,rtol=0,atol=1e-12*np.max(abs(M))):raise ValueError('Symmetric generalized mass required')
        try:L=np.linalg.cholesky(M)
        except np.linalg.LinAlgError as exc:raise ValueError('Positive definite generalized mass required') from exc
        self.mass=M.copy();self.jacobian=J.copy();self.joint_velocity=q.copy();self.link_velocity=v.copy()
        self.to_joint=np.linalg.solve(L.T,np.eye(3));self.velocity=L.T@q
        self.contact_map=J@self.to_joint
        self.observation_residual=v-J@q

    def joint_state(self,velocity):
        return self.to_joint@np.asarray(velocity,float)

    def contact_state(self,velocity):
        return self.contact_map@np.asarray(velocity,float)+self.observation_residual

    def generalized_impulse(self,world_impulse):
        return self.jacobian.T@np.asarray(world_impulse,float)

    def row(self,normal):
        normal=np.asarray(normal,float)
        if normal.shape!=(3,) or not np.isfinite(normal).all():raise ValueError('Finite contact normal required')
        return normal@self.contact_map,float(normal@self.observation_residual)
