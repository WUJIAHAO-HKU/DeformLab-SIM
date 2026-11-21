"""Candidate: same depth measurement operations with reusable dense buffers."""
import numpy as np

class ReusedDepthObserver:
    def __init__(self,side):
        self.side=side;self.shape=None;self.allocations=0
    def prepare(self,shape):
        if self.shape==shape:return
        h,w=shape
        if h<3 or w<3:raise ValueError('Image needs 3x3 normal neighborhood')
        self.shape=shape;self.allocations+=1
        self.yy,self.xx=np.indices(shape)
        self.d=np.empty(shape,float);self.safe=np.empty(shape,float)
        self.camera=np.empty((h,w,3),float);self.world=np.empty_like(self.camera)
        self.roi3=np.empty((h,w,3),bool);self.finite=np.empty(shape,bool);self.roimask=np.empty(shape,bool)
        self.mask=np.empty(shape,bool);self.interior=np.empty(shape,bool)
        inner=(h-2,w-2);vectors=inner+(3,)
        self.dx=np.empty(vectors,float);self.dy=np.empty(vectors,float);self.normal=np.empty(vectors,float)
        self.view=np.empty(vectors,float);self.squares=np.empty(vectors,float);self.delta=np.empty(vectors,float)
        for name in ['ln','lv','neighbor','cross_tmp','dot','denominator','cosine']:
            setattr(self,name,np.empty(inner,float))
        self.continuous=np.empty(inner,bool)
    def norm(self,v,out):
        np.multiply(v,v,out=self.squares)
        np.sum(self.squares,axis=-1,out=out)
        np.sqrt(out,out=out)
    def __call__(self,depth,calibration,roi=None,*,sample_time_s,sequence):
        depth=np.asarray(depth)
        if depth.shape!=(calibration.height,calibration.width):raise ValueError('Depth dimensions disagree with calibration')
        self.prepare(depth.shape)
        m=calibration.matrix();lo,hi=self.side.ROI.bounds()
        np.copyto(self.d,depth,casting='unsafe')
        d=self.d;self.finite[:]=np.isfinite(d)&(d>.01)&(d<5.)
        np.copyto(self.safe,d);self.safe[~self.finite]=0.
        np.multiply(self.xx+.5-calibration.cx,self.safe,out=self.camera[:,:,0]);self.camera[:,:,0]/=calibration.fx
        np.multiply(self.yy+.5-calibration.cy,self.safe,out=self.camera[:,:,1]);self.camera[:,:,1]/=calibration.fy
        self.camera[:,:,2]=self.safe
        np.matmul(self.camera,m[:3,:3].T,out=self.world);self.world+=m[:3,3]
        world=self.world
        np.greater_equal(world,lo,out=self.roi3);np.all(self.roi3,axis=-1,out=self.roimask)
        self.roimask&=self.finite
        np.less_equal(world,hi,out=self.roi3);self.roimask&=np.all(self.roi3,axis=-1)
        np.subtract(world[1:-1,2:],world[1:-1,:-2],out=self.dx)
        np.subtract(world[2:,1:-1],world[:-2,1:-1],out=self.dy)
        # Same multiply/subtract order as np.cross for 3D vectors.
        for component,j,k in [(0,1,2),(1,2,0),(2,0,1)]:
            np.multiply(self.dx[:,:,j],self.dy[:,:,k],out=self.normal[:,:,component])
            np.multiply(self.dx[:,:,k],self.dy[:,:,j],out=self.cross_tmp)
            np.subtract(self.normal[:,:,component],self.cross_tmp,out=self.normal[:,:,component])
        self.norm(self.normal,self.ln)
        np.subtract(calibration.matrix()[:3,3],world[1:-1,1:-1],out=self.view)
        self.norm(self.view,self.lv)
        np.multiply(self.normal,self.view,out=self.squares);np.sum(self.squares,axis=-1,out=self.dot);np.abs(self.dot,out=self.dot)
        np.multiply(self.ln,self.lv,out=self.denominator);self.cosine.fill(0.)
        np.divide(self.dot,self.denominator,out=self.cosine,where=(self.ln>1e-12)&(self.lv>0))
        self.continuous[:]=self.finite[1:-1,1:-1]
        for sy,sx in ((slice(1,-1),slice(2,None)),(slice(1,-1),slice(None,-2)),(slice(2,None),slice(1,-1)),(slice(None,-2),slice(1,-1))):
            np.subtract(world[sy,sx],world[1:-1,1:-1],out=self.delta)
            self.norm(self.delta,self.neighbor)
            self.continuous &= self.finite[sy,sx]&(self.neighbor<=.002)
        self.mask.fill(False);self.mask[1:-1,1:-1]=self.continuous&(self.cosine>=np.cos(np.pi/6))&self.roimask[1:-1,1:-1]
        mask=self.mask;self.interior.fill(False);self.interior[1:-1,1:-1]=True
        for y in (-1,0,1):
            for x in (-1,0,1):self.interior[1:-1,1:-1]&=mask[1+y:mask.shape[0]-1+y,1+x:mask.shape[1]-1+x]
        result=self.side._measure(world,self.interior,'depth_geometry',sample_time_s,sequence)
        result['normal_selection']=dict(max_angle_degrees=30,neighbor_distance_m=.002,erosion_pixels=1)
        return result
