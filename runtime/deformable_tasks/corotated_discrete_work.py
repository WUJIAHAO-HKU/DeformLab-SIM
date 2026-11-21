"""Objective two-point material force for a future coupled finite-strain solve.

Positive-volume corotated material only. This is a force operator, not a time
integrator or a native contact fix. The tiny-increment branch uses the smooth
midpoint limit and reports the independently evaluated work defect explicitly.
"""
import numpy as np
from .corotated_material_reference import cofactor


def density_difference(model,F0,J0,s0,Vh0,F1,J1,s1,Vh1,dC):
    """Energy secant without subtracting independently rounded singular values.

    For U_i=sqrt(C_i), U_1*dU+dU*U_0=dC. Solve this Sylvester
    equation in the two eigenbases, including repeated singular values.
    det(F1)-det(F0)=cof(Fmid):dF+det(dF)/4 is exact for 3x3 matrices.
    """
    V0=Vh0.transpose(0,2,1);rotation=Vh1@V0
    local=Vh1@dC@V0
    delta_root_trace=np.sum(local*rotation/(s1[:,:,None]+s0[:,None,:]),axis=(1,2))
    dF=F1-F0;Fmid=.5*(F0+F1)
    delta_J=np.sum(cofactor(Fmid)*dF,axis=(1,2))+.25*np.linalg.det(dF)
    return model.mu*(np.trace(dC,axis1=1,axis2=2)-2*delta_root_trace)+.5*model.lam*delta_J*(J1+J0-2)


def discrete_material_gradient(model,x0,x1):
    x0=np.asarray(x0,float);x1=np.asarray(x1,float)
    F0,J0,_,s0,Vh0=model.kinematics(x0);F1,J1,_,s1,Vh1=model.kinematics(x1)
    Fmid=.5*(F0+F1);dF=F1-F0
    Cmid=.5*(F0.transpose(0,2,1)@F0+F1.transpose(0,2,1)@F1)
    dC=Fmid.transpose(0,2,1)@dF+dF.transpose(0,2,1)@Fmid
    values,V=np.linalg.eigh(Cmid)
    if not np.all(values>0):raise ValueError('Positive metric required')
    Cinv=(V*(1/values)[:,None,:])@V.transpose(0,2,1)
    Cinvhalf=(V*(1/np.sqrt(values))[:,None,:])@V.transpose(0,2,1)
    Jmid=np.sqrt(np.prod(values,axis=1))
    G=model.mu*(np.eye(3)-Cinvhalf)+.5*model.lam*((Jmid-1)*Jmid)[:,None,None]*Cinv
    # Stable difference of the two densities, not an imposed work residual.
    difference=density_difference(model,F0,J0,s0,Vh0,F1,J1,s1,Vh1,dC)
    norm2=np.sum(dC*dC,axis=(1,2));scale=np.maximum(np.linalg.norm(Cmid,axis=(1,2)),1.)
    active=norm2>np.finfo(float).eps*scale**2
    correction=np.zeros_like(norm2)
    np.divide(difference-np.sum(G*dC,axis=(1,2)),norm2,out=correction,where=active)
    Gbar=G+correction[:,None,None]*dC
    # Symmetric Gbar gives zero torque about midpoint positions. The identity
    # dC=Fmid^T*dF+dF^T*Fmid makes Pbar:dF=Gbar:dC.
    Pbar=2*Fmid@Gbar
    gradient=model.scatter(model.volume[:,None,None]*model.local_gradient(Pbar))
    work=float(np.sum(gradient*(x1-x0)));energy_change=float(model.volume@difference)
    midpoint=.5*(x0+x1);center=midpoint.mean(0)
    diagnostics=dict(material_energy_change_J=energy_change,material_gradient_work_J=work,
        work_defect_J=work-energy_change,midpoint_limit_elements=int((~active).sum()),
        corrected_elements=int(active.sum()),net_internal_force_N=float(np.linalg.norm(gradient.sum(0))),
        midpoint_internal_torque_Nm=float(np.linalg.norm(np.cross(midpoint-center,gradient).sum(0))))
    return gradient,diagnostics
