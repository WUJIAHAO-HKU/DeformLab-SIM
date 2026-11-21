"""Replay actual authoring function against an in-memory USD-attribute contract."""
from pathlib import Path
import json,sys,types
import numpy as np
from completion_contract import completion
class Vec(tuple):
    def __new__(cls,*values):return tuple.__new__(cls,values)
class Attr:
    def __init__(self,value):self.value=value
    def Get(self):return self.value
    def Set(self,v):self.value=v
    def IsValid(self):return True
class Prim:
    def __init__(self,**attrs):self.attrs={k:Attr(v) for k,v in attrs.items()}
    def GetAttribute(self,key):return self.attrs.get(key)
    def GetAttributes(self):return []
    def GetPath(self):return getattr(self,'path','/test')
    def GetPointsAttr(self):return self.attrs['points']
    def GetExtentAttr(self):return self.attrs['extent']
class Stage:
    def __init__(self,points):
        self.prims={f'/World/Press/{n}':Prim(**{'xformOp:translate':Vec(0,0,.17)}) for n in ['Base','SlideX','SlideY','Tool','ToolDisplay']}
        self.prims['/World/Press/WorldAnchor']=Prim(anchor=Vec(0,0,.17))
        self.prims['/World/SoftA']=Prim(**{'points':points.copy(),'omniphysics:restShapePoints':points.copy(),'extent':np.array([points.min(0),points.max(0)])})
        self.prims['/World/Looks/GripPhysics']=Prim();self.prims['/World/SoftAPhysics']=Prim()
    def GetPrimAtPath(self,path):
        self.prims[path].path=path;return self.prims[path]
def test():
    proxy=types.SimpleNamespace(UsdGeom=object(),UsdPhysics=types.SimpleNamespace(FixedJoint=lambda p:types.SimpleNamespace(GetLocalPos0Attr=lambda:p.attrs['anchor'])),Gf=types.SimpleNamespace(Vec3f=Vec),Vt=types.SimpleNamespace(Vec3fArray=types.SimpleNamespace(FromNumpy=lambda x:np.array(x,copy=True))))
    sys.modules['pxr']=proxy
    from side_push_fixture import relocate_fixture
    original=np.array([[-.02,-.02,.12],[.02,-.02,.12],[-.02,.02,.12],[-.02,-.02,.16]],dtype=np.float32)
    determinant=lambda p:float(np.linalg.det(p[1:].astype(float)-p[0].astype(float)))
    results=[]
    for scale in [.75,1.,1.25]:
        for offset in [-.0025,0.,.0025]:
            stage=Stage(original);visual=Prim(points=original.copy(),extent=np.array([original.min(0),original.max(0)]))
            b=dict(rest=original.copy(),path='/World/SoftA',visual=visual)
            runtime=types.SimpleNamespace(steps=0,movers=[(object(),np.array([0,0,.17]),'/World/Press')],bodies=[b])
            rec=relocate_fixture(stage,runtime,offset,scale)
            prim=stage.prims[b['path']];current=np.asarray(prim.GetAttribute('points').Get());rest=np.asarray(prim.GetAttribute('omniphysics:restShapePoints').Get())
            assert np.array_equal(current,rest) and np.array_equal(current,visual.GetPointsAttr().Get())
            assert np.allclose(np.ptp(rest,axis=0),[.04,.04*scale,.04],atol=1e-8,rtol=0)
            assert np.isclose(determinant(rest)/determinant(original),scale,rtol=2e-7,atol=0)
            F=(current[1:]-current[0]).T@np.linalg.inv((rest[1:]-rest[0]).T)
            assert np.allclose(F,np.eye(3),atol=1e-7,rtol=0)
            assert np.array_equal(prim.GetAttribute('extent').Get(),visual.GetExtentAttr().Get())
            assert rec['reference_shape_authoring'][0]['authored_current_equals_reference']
            results.append(dict(y_scale=scale,offset=offset,stress_free_reference=True))
    config=dict(duration_s=3.2,solver=dict(hz=960));good=dict(status='completed_pending_exit',exit_code=0,errors=[],completed_steps=3072)
    assert completion(good,0,config)['passed']
    invalid=[{},dict(status='failed',exit_code=1,errors=[],error='rest mismatch'),dict(good,completed_steps=3071),dict(good,exit_code=1),dict(good,errors=['cleanup'])]
    for receipt in invalid:assert not completion(receipt,0,config)['passed']
    assert not completion(good,124,config)['passed']
    return dict(passed=True,authoring_cases=results,raw_zero_failed_semantic_rejected=True,scope='Actual authoring function with stub USD attributes and tetrahedral reference-gradient/volume check; native GPU cook still requires new smoke')
if __name__=='__main__':print(json.dumps(test(),indent=2))
