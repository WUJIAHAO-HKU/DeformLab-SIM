"""Read-only outer-step nodal endpoints, separating native solve and split damping."""
import json
from pathlib import Path
import numpy as np


class NodalMotionRecorder:
    def __init__(self, destination, mass, hz, nsteps):
        self.destination=Path(destination);self.mass=np.asarray(mass,dtype=np.float64).copy()
        self.hz=float(hz);self.nsteps=int(nsteps);self.rows=[];self.positions=[];self.velocities=[]
        if self.mass.ndim!=1 or not len(self.mass) or not np.isfinite(self.mass).all() or np.any(self.mass<=0):
            raise ValueError('Finite positive nodal masses required')
        if not np.isfinite(self.hz) or self.hz<=0 or self.nsteps<1:raise ValueError('Invalid horizon')

    def capture(self, step, phase, positions, velocities):
        expected=(0,'initial') if not self.rows else ((len(self.rows)+1)//2,'native' if len(self.rows)%2 else 'damped')
        if (step,phase)!=expected or step>self.nsteps:raise ValueError('Nodal snapshots must follow initial/native/damped order')
        p=np.asarray(positions);v=np.asarray(velocities)
        if p.shape!=(len(self.mass),3) or v.shape!=p.shape or not np.isfinite(p).all() or not np.isfinite(v).all():
            raise ValueError('Finite matching three-dimensional nodal states required')
        # Preserve observed float32 bits; reject a silent cast of other input precision.
        if p.dtype!=np.float32 or v.dtype!=np.float32:raise ValueError('Expected native float32 nodal state')
        self.rows.append((step,phase));self.positions.append(p.copy());self.velocities.append(v.copy())

    def finish(self):
        complete=len(self.rows)==1+2*self.nsteps
        if not self.rows:raise ValueError('Missing initial state')
        np.savez_compressed(self.destination/'nodal-motion.npz',
            steps=np.asarray([r[0] for r in self.rows],dtype=np.int64),
            phases=np.asarray([r[1] for r in self.rows]),
            positions=np.asarray(self.positions),velocities=np.asarray(self.velocities),mass_kg=self.mass,
            time_s=np.asarray([r[0]/self.hz for r in self.rows]))
        result=dict(schema_version=1,enabled=True,complete=complete,records=len(self.rows),expected_records=1+2*self.nsteps,
                    hz=self.hz,nodes=len(self.mass),requested_steps=self.nsteps,
                    scope='Read-only float32 nodal states at initial time, immediately after native solve/admissibility and after split material damping. No inner-step velocities or contact impulses. Intrusive readbacks are not performance evidence.')
        (self.destination/'nodal-motion.json').write_text(json.dumps(result,indent=2)+'\n')
        return result


def momentum_state(positions, velocities, mass):
    """World-origin and COM angular momentum; retain translation and rotation."""
    p=np.asarray(positions,dtype=float);v=np.asarray(velocities,dtype=float);m=np.asarray(mass,dtype=float)
    if p.shape!=v.shape or p.shape[-2:]!=(len(m),3):raise ValueError('State/mass shape mismatch')
    if not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(m).all() or np.any(m<=0):raise ValueError('Invalid state')
    com=np.sum(p*m[:,None],axis=-2)/m.sum();linear=np.sum(v*m[:,None],axis=-2)
    angular_world=np.sum(np.cross(p,v)*m[:,None],axis=-2)
    angular_com=angular_world-np.cross(com,linear)
    kinetic=.5*np.sum(m*np.sum(v*v,axis=-1),axis=-1)
    return dict(com=com,linear=linear,angular_world=angular_world,angular_com=angular_com,kinetic=kinetic)
