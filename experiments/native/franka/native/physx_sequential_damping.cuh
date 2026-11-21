// Version-locked experiment. Original PhysX computes the rotation; this update
// consumes cached partition inputs. No root/body velocity target is imposed.
namespace sd {
__device__ double det(const double a[3][3]) {
    return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
}
__device__ void deformation(double p[4][3], double q[3][3], double f[3][3]) {
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){f[i][j]=0;for(int k=0;k<3;++k)f[i][j]+=(p[k+1][i]-p[0][i])*q[k][j];}
}
__device__ void gradient(double t[3][3],double q[3][3],double g[4][3]) {
    for(int j=0;j<3;++j){g[0][j]=0;for(int i=0;i<3;++i){g[i+1][j]=0;for(int k=0;k<3;++k)g[i+1][j]+=t[j][k]*q[i][k];g[0][j]-=g[i+1][j];}}
}
__device__ void solve(double c,double g[4][3],double v[4][3],double w[4],double alpha,double gamma,double invdt,double dx[4][3]) {
    double s=0,rate=0;for(int i=0;i<4;++i)for(int j=0;j<3;++j){s+=w[i]*g[i][j]*g[i][j];rate+=g[i][j]*v[i][j];}
    double dl=-(c+gamma*rate)/((1+gamma*invdt)*s+alpha);
    for(int i=0;i<4;++i)for(int j=0;j<3;++j)dx[i][j]=w[i]*g[i][j]*dl;
}
// One coupled linearization of ARAP and volume, using a shared incoming
// velocity. Row scaling yields an SPD 2x2 system; no damping coefficient fit.
__device__ bool solve_coupled(double C[2],double G[2][4][3],double v[4][3],double w[4],double alpha[2],double gamma[2],double invdt,double dx[4][3]) {
    double S[2][2]={{0,0},{0,0}},rhs[2]={-C[0],-C[1]};
    for(int a=0;a<2;++a){
        double rate=0;
        for(int i=0;i<4;++i)for(int j=0;j<3;++j){
            rate+=G[a][i][j]*v[i][j];
            for(int b=0;b<2;++b)S[a][b]+=w[i]*G[a][i][j]*G[b][i][j];
        }
        double scale=1+gamma[a]*invdt;
        rhs[a]=-(C[a]+gamma[a]*rate)/scale;S[a][a]+=alpha[a]/scale;
    }
    double determinant=S[0][0]*S[1][1]-S[0][1]*S[1][0];
    if(!(determinant>0) || !isfinite(determinant))return false;
    double l0=(rhs[0]*S[1][1]-rhs[1]*S[0][1])/determinant;
    double l1=(rhs[1]*S[0][0]-rhs[0]*S[1][0])/determinant;
    for(int i=0;i<4;++i)for(int j=0;j<3;++j)dx[i][j]=w[i]*(G[0][i][j]*l0+G[1][i][j]*l1);
    return true;
}
// Opt-in objective corotational reference: positive-determinant F only.
// Newton polar iteration is established numerical algebra, not a novel method.
__device__ bool polar_rotation(const double F[3][3],double R[3][3]) {
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)R[i][j]=F[i][j];
    for(int it=0;it<8;++it){
        double d=det(R),next[3][3];if(!(d>1e-12) || !isfinite(d))return false;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            double co=R[(i+1)%3][(j+1)%3]*R[(i+2)%3][(j+2)%3]-R[(i+1)%3][(j+2)%3]*R[(i+2)%3][(j+1)%3];
            next[i][j]=.5*(R[i][j]+co/d);
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)R[i][j]=next[i][j];
    }
    double err=0;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){double a=0;for(int k=0;k<3;++k)a+=R[k][i]*R[k][j];err=fmax(err,fabs(a-(i==j?1.:0.)));}
    return isfinite(err) && err<1e-10 && det(R)>0;
}
// Experimental strain-scaled stopping rule. Starting at F preserves the
// Newton polar branch; orthogonality alone would not validate a cached rotation.
// The absolute floor acknowledges double roundoff, not an exact torque bound.
__device__ bool adaptive_polar_rotation(const double F[3][3],double R[3][3],int* iterations=nullptr) {
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
        if(!isfinite(F[i][j]))return false;
        R[i][j]=F[i][j];
    }
    for(int it=0;it<8;++it){
        double d=det(R),next[3][3];if(!(d>1e-12) || !isfinite(d))return false;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            double co=R[(i+1)%3][(j+1)%3]*R[(i+2)%3][(j+2)%3]-R[(i+1)%3][(j+2)%3]*R[(i+2)%3][(j+1)%3];
            next[i][j]=.5*(R[i][j]+co/d);
        }
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)R[i][j]=next[i][j];
        double err=0,c2=1e-18;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){
            double a=0;for(int k=0;k<3;++k)a+=R[k][i]*R[k][j];
            if(!isfinite(a) || !isfinite(R[i][j]))return false;
            err=fmax(err,fabs(a-(i==j?1.:0.)));
            double strain=F[i][j]-R[i][j];c2+=strain*strain;
        }
        if(iterations)*iterations=it+1;
        if(err<=fmax(8e-15,fmin(1e-10,1e-8*sqrt(c2))))return det(R)>0;
    }
    return false;
}
}
extern "C" __global__ void cache_partition_inputs(char* sb,float4* cache,unsigned part) {
    auto prefix=field<unsigned*>(sb,208);unsigned tet=(part?prefix[part-1]:0)+blockIdx.x*blockDim.x+threadIdx.x;
    if(tet>=prefix[part])return;
    unsigned nt=field<unsigned>(sb,376);auto pull=field<uint4*>(sb,400);
    uint4 ids=pull[tet];ids.x&=0x7fffffff;unsigned* id=&ids.x;
    auto p=field<float4*>(sb,96);auto v=field<float4*>(sb,88);
    auto cp=field<float4*>(sb,184);auto cv=field<float4*>(sb,176);
    for(int k=0;k<4;++k){bool root=id[k]<0x7fffffff;cache[8*tet+k]=root?p[id[k]]:cp[tet+k*nt];cache[8*tet+4+k]=root?v[id[k]]:cv[tet+k*nt];}
}
extern "C" __global__ void update_partition_damping(char* sb,float4* cache,unsigned part,float invdt,char* materials,int sequential,unsigned* rejected) {
    auto prefix=field<unsigned*>(sb,208);unsigned tet=(part?prefix[part-1]:0)+blockIdx.x*blockDim.x+threadIdx.x;
    if(tet>=prefix[part])return;
    auto mid=field<unsigned short*>(sb,288);char* mat=materials+32*mid[tet];
    double E=*reinterpret_cast<float*>(mat),nu=*reinterpret_cast<float*>(mat+4),gamma=*reinterpret_cast<float*>(mat+12);
    if(*reinterpret_cast<unsigned short*>(mat+16)!=0 || !(E>0 && nu>0 && nu<.5 && gamma>=0 && invdt>0)){atomicAdd(rejected,1u);return;}
    double p[4][3],v[4][3],w[4];
    for(int i=0;i<4;++i){float4 a=cache[8*tet+i],b=cache[8*tet+4+i];w[i]=b.w;
        if(!(w[i]>0) || a.w!=b.w){atomicAdd(rejected,1u);return;}
        p[i][0]=a.x;p[i][1]=a.y;p[i][2]=a.z;v[i][0]=b.x;v[i][1]=b.y;v[i][2]=b.z;}
    auto rest=field<float*>(sb,136)+288*(tet/32);unsigned slot=tet%32;double q[3][3];
    for(int k=0;k<9;++k)q[k%3][k/3]=rest[k<4?4*slot+k:k<8?128+4*slot+k-4:256+slot];
    double qdet=sd::det(q);if(!(qdet>0)){atomicAdd(rejected,1u);return;}
    float4 rot=field<float4*>(sb,168)[tet];double x=rot.x,y=rot.y,z=rot.z,rw=rot.w;
    double R[3][3]={{1-2*(y*y+z*z),2*(x*y-z*rw),2*(x*z+y*rw)},
        {2*(x*y+z*rw),1-2*(x*x+z*z),2*(y*z-x*rw)},
        {2*(x*z-y*rw),2*(y*z+x*rw),1-2*(x*x+y*y)}};
    double F[3][3],T[3][3],g[4][3],dx1[4][3],dx2[4][3],p2[4][3],v2[4][3];
    sd::deformation(p,q,F);
    if((sequential==2 || sequential==4) && !sd::polar_rotation(F,R)){atomicAdd(rejected,1u);return;}
    if(sequential==3 && !sd::adaptive_polar_rotation(F,R)){atomicAdd(rejected,1u);return;}
    double c2=1e-18;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){T[i][j]=F[i][j]-R[i][j];c2+=T[i][j]*T[i][j];}
    double c=sqrt(c2);for(int i=0;i<3;++i)for(int j=0;j<3;++j)T[i][j]/=c;
    double mu=E/(2*(1+nu)),lambda=E*nu/((1+nu)*(1-2*nu)),volume=1/(6*qdet),h2=double(invdt)*invdt;
    sd::gradient(T,q,g);
    if(sequential==4){
        double G[2][4][3],C[2]={c,sd::det(F)-1},alpha[2]={h2/(2*mu*volume),h2/(lambda*volume)},gammas[2]={gamma,gamma*2*mu/lambda};
        for(int i=0;i<4;++i)for(int j=0;j<3;++j)G[0][i][j]=g[i][j];
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)T[i][j]=F[(i+1)%3][(j+1)%3]*F[(i+2)%3][(j+2)%3]-F[(i+1)%3][(j+2)%3]*F[(i+2)%3][(j+1)%3];
        sd::gradient(T,q,G[1]);
        if(!sd::solve_coupled(C,G,v,w,alpha,gammas,invdt,dx1)){atomicAdd(rejected,1u);return;}
        for(int i=0;i<4;++i)for(int j=0;j<3;++j)dx2[i][j]=0;
    }else{
    sd::solve(c,g,v,w,h2/(2*mu*volume),gamma,invdt,dx1);
    for(int i=0;i<4;++i)for(int j=0;j<3;++j){p2[i][j]=p[i][j]+dx1[i][j];v2[i][j]=v[i][j]+(sequential?dx1[i][j]*invdt:0);}
    sd::deformation(p2,q,F);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)T[i][j]=F[(i+1)%3][(j+1)%3]*F[(i+2)%3][(j+2)%3]-F[(i+1)%3][(j+2)%3]*F[(i+2)%3][(j+1)%3];
    sd::gradient(T,q,g);sd::solve(sd::det(F)-1,g,v2,w,h2/(lambda*volume),gamma*2*mu/lambda,invdt,dx2);
    }
    float4 op[4],ov[4];
    for(int i=0;i<4;++i){op[i]=cache[8*tet+i];ov[i]=cache[8*tet+4+i];for(int j=0;j<3;++j){
        (&op[i].x)[j]=float(p[i][j]+dx1[i][j]+dx2[i][j]);(&ov[i].x)[j]=float(v[i][j]+(dx1[i][j]+dx2[i][j])*invdt);
        if(!isfinite((&op[i].x)[j]) || !isfinite((&ov[i].x)[j])){atomicAdd(rejected,1u);return;}}}
    uint4 remap=field<uint4*>(sb,192)[tet];auto cp=field<float4*>(sb,184);auto cv=field<float4*>(sb,176);
    for(int i=0;i<4;++i){cp[(&remap.x)[i]]=op[i];cv[(&remap.x)[i]]=ov[i];}
}
