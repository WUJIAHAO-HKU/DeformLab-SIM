// Experimental low-overhead stopping variant; included after namespace sd.
namespace sd {
__device__ bool increment_polar_rotation(const double F[3][3],double R[3][3],int* iterations=nullptr) {
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){if(!isfinite(F[i][j]))return false;R[i][j]=F[i][j];}
    for(int it=0;it<8;++it){
        double d=det(R),next[3][3];if(!(d>1e-12)||!isfinite(d))return false;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            double co=R[(i+1)%3][(j+1)%3]*R[(i+2)%3][(j+2)%3]-R[(i+1)%3][(j+2)%3]*R[(i+2)%3][(j+1)%3];
            next[i][j]=.5*(R[i][j]+co/d);
        }
        // In exact arithmetic N^T N-I = D^T D for N=(R+R^-T)/2,
        // D=N-R. Thus ||D||_F^2 bounds every orthogonality residual entry.
        // Check every two iterations to reduce control and residual overhead.
        double bound=0,c2=1e-18;
        if(it%2==1)for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            double delta=next[i][j]-R[i][j],strain=F[i][j]-next[i][j];
            bound+=delta*delta;c2+=strain*strain;
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)R[i][j]=next[i][j];
        if(iterations)*iterations=it+1;
        if(it%2==1){
            if(!isfinite(bound)||!isfinite(c2))return false;
            if(bound<=fmax(8e-15,fmin(1e-10,1e-8*sqrt(c2))))return det(R)>0;
        }
    }
    return false;
}
}
