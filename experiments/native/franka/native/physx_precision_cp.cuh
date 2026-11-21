// Full-precision CP chains. Uses established sd math from physx_sequential_damping.cuh.
extern "C" __global__ void update_partition_precision(char* sb,double4* cache,unsigned part,float invdt,char* materials,int sequential,unsigned* rejected,double4* precise_p,double4* precise_v) {
    auto prefix=field<unsigned*>(sb,208);unsigned tet=(part?prefix[part-1]:0)+blockIdx.x*blockDim.x+threadIdx.x;
    if(tet>=prefix[part])return;
    auto mid=field<unsigned short*>(sb,288);char* mat=materials+32*mid[tet];
    double E=*reinterpret_cast<float*>(mat),nu=*reinterpret_cast<float*>(mat+4),gamma=*reinterpret_cast<float*>(mat+12);
    if(*reinterpret_cast<unsigned short*>(mat+16)!=0 || !(E>0 && (nu>0 || (nu==0 && gamma==0)) && nu<.5 && gamma>=0 && invdt>0)){atomicAdd(rejected,1u);return;}
    double p[4][3],v[4][3],w[4];
    for(int i=0;i<4;++i){double4 a=cache[8*tet+i],b=cache[8*tet+4+i];w[i]=b.w;
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
    if(sequential==2 && !sd::polar_rotation(F,R)){atomicAdd(rejected,1u);return;}
    if(sequential==3 && !sd::adaptive_polar_rotation(F,R)){atomicAdd(rejected,1u);return;}
    double c2=1e-18;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){T[i][j]=F[i][j]-R[i][j];c2+=T[i][j]*T[i][j];}
    double c=sqrt(c2);for(int i=0;i<3;++i)for(int j=0;j<3;++j)T[i][j]/=c;
    double mu=E/(2*(1+nu)),lambda=E*nu/((1+nu)*(1-2*nu)),volume=1/(6*qdet),h2=double(invdt)*invdt;
    sd::gradient(T,q,g);sd::solve(c,g,v,w,h2/(2*mu*volume),gamma,invdt,dx1);
    for(int i=0;i<4;++i)for(int j=0;j<3;++j){p2[i][j]=p[i][j]+dx1[i][j];v2[i][j]=v[i][j]+(sequential?dx1[i][j]*invdt:0);}
    sd::deformation(p2,q,F);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)T[i][j]=F[(i+1)%3][(j+1)%3]*F[(i+2)%3][(j+2)%3]-F[(i+1)%3][(j+2)%3]*F[(i+2)%3][(j+1)%3];
    // At nu=0 and zero damping, lambda=0: volumetric energy is absent.
    // Avoid the undefined 0/0 damping ratio; retain the old positive-nu path.
    if(lambda==0){for(int i=0;i<4;++i)for(int j=0;j<3;++j)dx2[i][j]=0;}
    else {sd::gradient(T,q,g);sd::solve(sd::det(F)-1,g,v2,w,h2/(lambda*volume),gamma*2*mu/lambda,invdt,dx2);}
    double4 op[4],ov[4];
    for(int i=0;i<4;++i){op[i]=cache[8*tet+i];ov[i]=cache[8*tet+4+i];for(int j=0;j<3;++j){
        (&op[i].x)[j]=(p[i][j]+dx1[i][j]+dx2[i][j]);(&ov[i].x)[j]=(v[i][j]+(dx1[i][j]+dx2[i][j])*invdt);
        if(!isfinite((&op[i].x)[j]) || !isfinite((&ov[i].x)[j])){atomicAdd(rejected,1u);return;}}}
    uint4 remap=field<uint4*>(sb,192)[tet];auto cp=field<float4*>(sb,184);auto cv=field<float4*>(sb,176);
    for(int i=0;i<4;++i){unsigned j=(&remap.x)[i];if(j>=8*field<unsigned>(sb,376)){atomicAdd(rejected,1u);return;}
        precise_p[j]=op[i];precise_v[j]=ov[i];
        cp[j]=make_float4(op[i].x,op[i].y,op[i].z,op[i].w);cv[j]=make_float4(ov[i].x,ov[i].y,ov[i].z,ov[i].w);}
}
