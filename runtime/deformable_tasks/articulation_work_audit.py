"""Read-only work audit for the verified Cartesian, fixed-base coupon fixture.

Projected joint force is identified with net drive effort ONLY after checking
zero joint friction/armature, inactive limits, zero gravity and no applied
effort. This is not a general robot motor-power estimator.
"""
import json
import numpy as np


class ArticulationWorkAudit:
    def __init__(self, runtime, mode, destination):
        if mode not in ('coupon', 'empty_motion'):
            raise ValueError('Articulation work requires coupon or empty_motion')
        self.runtime = runtime
        self.destination = destination
        self.drive = runtime.drives[0]
        self.tool = runtime.movers[0][0]
        self.indices = [self.drive.dof_names.index('Drive'+a) for a in 'XYZ']
        self.links = [self.drive.link_names.index(n) for n in ('SlideX', 'SlideY', 'Tool')]
        self.rows = []
        self.initial = self.capture()
        expected = np.diag([2.2, 1.2, .2])
        # Preserve the float32 tensor matrix, including tiny off-axis entries
        # from authored joint-frame rotations; use independent authored masses.
        self.mass = expected
        (destination/'articulation-bind-state.json').write_text(json.dumps(
            {k:v.tolist() for k,v in self.initial.items()},indent=2)+'\n')
        if np.linalg.norm(self.initial['mass']-expected) > 2e-6*np.linalg.norm(expected):
            raise RuntimeError('Unexpected initial Cartesian mass matrix')
        if not np.allclose(self.initial['v'], self.initial['tool_v'], atol=2e-6):
            raise RuntimeError('Unexpected Cartesian velocity mapping')
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend('tensor', raise_on_fallback=True, raise_on_unsupported=True):
            friction = [x.numpy()[0, self.indices] for x in self.drive.get_dof_friction_properties()]
            armature = self.drive.get_dof_armatures().numpy()[0, self.indices]
            limits = np.stack([x.numpy()[0, self.indices] for x in self.drive.get_dof_limits()],axis=1)
            gains = [x.numpy()[0, self.indices] for x in self.drive.get_dof_gains()]
        if any(np.max(abs(x)) > 1e-10 for x in [*friction, armature]):
            raise RuntimeError('Passive joint effort would confound drive work')
        self.limits = limits
        self.metadata = dict(mode=mode, dof_order=['DriveX','DriveY','DriveZ'],
            link_names=self.drive.link_names, generalized_mass_kg=self.mass.tolist(),
            friction=[x.tolist() for x in friction], armature=armature.tolist(),
            limits=limits.tolist(), gains=[x.tolist() for x in gains],
            scope='World-aligned fixed-base Cartesian prismatic fixture; zero gravity, zero joint friction/armature, inactive limits. Measured projected force is net active drive effort. Full-frame midpoint impulse work, not intra-step actuator quadrature or motor electrical energy.')

    def capture(self):
        from isaacsim.core.experimental.utils.backend import use_backend
        with use_backend('tensor', raise_on_fallback=True, raise_on_unsupported=True):
            q = self.drive.get_dof_positions().numpy()[0, self.indices].astype(float)
            v = self.drive.get_dof_velocities().numpy()[0, self.indices].astype(float)
            target = self.drive.get_dof_position_targets().numpy()[0, self.indices].astype(float)
            tau = self.drive.get_dof_projected_joint_forces().numpy()[0, self.indices].astype(float)
            effort = self.drive.get_dof_efforts().numpy()[0, self.indices].astype(float)
            mass = self.drive.get_mass_matrices().numpy()[0].astype(float)[np.ix_(self.indices, self.indices)]
            forces, torques = self.drive.get_link_incoming_joint_force(link_indices=self.links)
            force = forces.numpy()[0].astype(float)
            torque = torques.numpy()[0].astype(float)
            tool_v, tool_w = self.tool.get_velocities()
            tool_v = tool_v.numpy()[0].astype(float)
            tool_w = tool_w.numpy()[0].astype(float)
            orientation = self.tool.get_world_poses()[1].numpy()[0].astype(float)
        return dict(q=q, v=v, target=target, tau=tau, applied_effort=effort, mass=mass,
            incoming_force=force, incoming_torque=torque, tool_v=tool_v, tool_w=tool_w, orientation=orientation)

    def after_step(self):
        row = self.capture()
        if np.linalg.norm(row['mass']-self.mass) > 2e-6*np.linalg.norm(self.mass):
            raise RuntimeError('Cartesian mass check failed: '+repr(row['mass'].tolist()))
        if np.any(row['q'] <= self.limits[:,0]+.01) or np.any(row['q'] >= self.limits[:,1]-.01):
            raise RuntimeError('Joint limit active in drive work audit')
        if np.max(abs(row['applied_effort'])) > 1e-10:
            raise RuntimeError('Additional applied effort in drive work audit')
        if np.max(abs(row['tool_w'])) > 1e-5 or not np.allclose(abs(row['orientation']), [1,0,0,0], atol=1e-6):
            raise RuntimeError('Rotating fixture needs full spatial work audit')
        if not np.allclose(row['v'], row['tool_v'], atol=2e-6):
            raise RuntimeError('Cartesian velocity mapping failed')
        row['step'] = self.runtime.steps
        self.rows.append(row)

    def finish(self):
        data = {key:np.asarray([self.initial[key]]+[r[key] for r in self.rows]) for key in self.initial}
        data['steps'] = np.asarray([0]+[r['step'] for r in self.rows])
        np.savez_compressed(self.destination/'articulation-work-states.npz', **data)
        (self.destination/'articulation-work-fixture.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
        return dict(samples=len(self.rows), scope=self.metadata['scope'])
