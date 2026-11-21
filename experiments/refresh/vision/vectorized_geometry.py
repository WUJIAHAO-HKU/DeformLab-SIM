"""Same three-term Euclidean norm, explicitly ordered NumPy operations."""
import numpy as np
from reused_geometry import ReusedDepthObserver

class VectorizedReusedDepthObserver(ReusedDepthObserver):
    def norm(self,v,out):
        # np.linalg.norm(..., axis=-1) is sqrt(sum(v*v)) here, with axis length3.
        # Preserve ((x*x + y*y) + z*z); no einsum/FMA/reassociation or dtype change.
        np.multiply(v[:,:,0],v[:,:,0],out=out)
        np.multiply(v[:,:,1],v[:,:,1],out=self.cross_tmp)
        np.add(out,self.cross_tmp,out=out)
        np.multiply(v[:,:,2],v[:,:,2],out=self.cross_tmp)
        np.add(out,self.cross_tmp,out=out)
        np.sqrt(out,out=out)
