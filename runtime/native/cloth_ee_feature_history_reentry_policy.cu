// Owned persistent single-cloth EE feature history. No native ABI hook yet.
// Keys use four canonical 16-bit node IDs, not transient contact-row/face IDs.
#include "cloth_finite_history_birth_v2.cu"
struct FH { unsigned long long key; int side; unsigned epoch; unsigned state; unsigned seen; };
// state: 0 empty, 1 ready, 2 safely separated, 3 near unsupported geometry,
// 4 ambiguous initialization, 5 crossed, 6 invalid input/update.
__device__ unsigned long long fh_key(uint4 r,unsigned nodes){
 unsigned a=r.x,b=r.y,c=r.z,d=r.w;
 if(nodes>65536||a>=nodes||b>=nodes||c>=nodes||d>=nodes||a==b||c==d||a==c||a==d||b==c||b==d)return 0;
 if(a>b){unsigned t=a;a=b;b=t;}if(c>d){unsigned t=c;c=d;d=t;}
 if(a>c||(a==c&&b>d)){unsigned t=a;a=c;c=t;t=b;b=d;d=t;}
 return (static_cast<unsigned long long>(a)<<48)|(static_cast<unsigned long long>(b)<<32)|(static_cast<unsigned long long>(c)<<16)|d;
}
__device__ unsigned fh_hash(unsigned long long k){k^=k>>33;k*=0xff51afd7ed558ccdull;k^=k>>33;k*=0xc4ceb9fe1a85ec53ull;k^=k>>33;return unsigned(k)^unsigned(k>>32);}
extern "C" __global__ void cloth_ee_history_resolve(const uint4* rows,unsigned count,unsigned nodes,FH* history,unsigned capacity,unsigned epoch,unsigned* slots,unsigned* errors){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;slots[i]=~0u;
 if(!capacity||(capacity&(capacity-1))||!epoch){atomicAdd(errors,1);return;}
 unsigned long long key=fh_key(rows[i],nodes);if(!key){atomicAdd(errors,1);return;}
 unsigned home=fh_hash(key)&(capacity-1);
 for(unsigned j=0;j<capacity;++j){unsigned at=(home+j)&(capacity-1);unsigned long long old=atomicCAS(&history[at].key,0ull,key);
  if(old==0||old==key){slots[i]=at;atomicMax(&history[at].seen,epoch);return;}
 }
 atomicAdd(errors,1);
}
__device__ double fh_point_segment(SV p,SV a,SV b){SV e=sub(b,a),v=sub(p,a);double den=dot(e,e),t=den>0?fmin(1.,fmax(0.,dot(v,e)/den)):0;SV r=sub(v,mul(e,t));return dot(r,r);}
__device__ double fh_segment_distance(const float4* p){
 SV a=xyz(p[0]),b=xyz(p[1]),c=xyz(p[2]),d=xyz(p[3]);double best=fmin(fmin(fh_point_segment(a,c,d),fh_point_segment(b,c,d)),fmin(fh_point_segment(c,a,b),fh_point_segment(d,a,b)));
 SV u=sub(b,a),v=sub(d,c),w=sub(a,c),n=cross(u,v);double nn=dot(n,n);
 if(nn>1e-30*dot(u,u)*dot(v,v)){double s=dot(cross(v,w),n)/nn,t=dot(cross(u,w),n)/nn;if(s>=0&&s<=1&&t>=0&&t<=1){SV r=sub(add(w,mul(u,s)),mul(v,t));best=fmin(best,dot(r,r));}}
 return sqrt(fmax(0.,best));
}
extern "C" { __device__ unsigned cloth_history_reentry_enabled=1; __device__ unsigned long long cloth_history_birth_counts[3]; __device__ unsigned long long cloth_history_surface_release_counts[2]; __device__ unsigned long long cloth_history_reentry_counts[3]; }
extern "C" __global__ void cloth_ee_history_update(FH* history,unsigned capacity,const float4* previous,const float4* current,unsigned nodes,unsigned epoch,double release_distance,unsigned* errors){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=capacity)return;FH& h=history[i];if(!h.key)return;
 if(nodes>65536||!epoch||epoch<=h.epoch||!isfinite(release_distance)||release_distance<=0){h.state=6;atomicAdd(errors,1);return;}
 unsigned ids[4]={unsigned(h.key>>48),unsigned((h.key>>32)&65535),unsigned((h.key>>16)&65535),unsigned(h.key&65535)};
 float4 old[4],cur[4];for(int j=0;j<4;++j){if(ids[j]>=nodes){h.state=6;atomicAdd(errors,1);return;}old[j]=previous[ids[j]];cur[j]=current[ids[j]];if(!finite4(old[j])||!finite4(cur[j])){h.state=6;atomicAdd(errors,1);return;}}
 h.epoch=epoch;SV oldn,n;double oldgap,gap,os,ot,s,t;bool valid=geometry(cur,n,gap,s,t),oldvalid=geometry(old,oldn,oldgap,os,ot);double distance=fh_segment_distance(cur);
 // Keep every occupied feature observable, including rows absent this generation.
 // Never release an interior feature merely because it is far on its crossed side.
 // release_distance is four rest offsets; half is the two-surface contact envelope.
 // A finite feature outside its interior support cannot retain an oriented
 // interior plane after the two finite segments are beyond that envelope.
 // Crossed interior features are still retained, however far behind the plane.
 bool envelope_release=!valid&&distance>0.5*release_distance+1e-12;
 if(envelope_release&&distance<=release_distance&&(h.side||h.state!=2)){atomicAdd(cloth_history_surface_release_counts+1,1ull);if(h.side)atomicAdd(cloth_history_surface_release_counts,1ull);}
 if((distance>release_distance&&(!valid||h.side==0||gap*h.side>0))||envelope_release){h.side=0;h.state=2;return;}
 if(!valid){h.state=3;return;}
 // Both a new feature and a released feature may have moved around finite
 // endpoints since the frame root. Opposite infinite-plane signs alone do
 // not prove a finite-edge crossing. Only a certified nonintersecting chord
 // changes the initialization side; unresolved/actual crossings retain the
 // existing root-side correction. Retained nonzero histories are unchanged.
 if(h.side==0){if(oldvalid&&fabs(oldgap)>1e-12){h.side=oldgap>0?1:-1;if(cloth_history_reentry_enabled&&oldgap*gap<0){atomicAdd(cloth_history_reentry_counts,1ull);int finite_side=fh_finite_birth_counted(old,cur);if(finite_side)h.side=finite_side;atomicAdd(cloth_history_reentry_counts+(finite_side?1:2),1ull);}}else if(!oldvalid){atomicAdd(cloth_history_birth_counts,1ull);h.side=fh_finite_birth_counted(old,cur);atomicAdd(cloth_history_birth_counts+(h.side?1:2),1ull);}if(!h.side){h.state=4;return;}}
 h.state=gap*h.side<0?5:1;
}
extern "C" __global__ void cloth_ee_history_normal(const FH* history,const unsigned* slots,unsigned count,const float4* positions,unsigned nodes,unsigned capacity,unsigned epoch,double separation,double h,float4* corrections,unsigned* states,unsigned* errors){
 unsigned row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=count)return;unsigned at=slots[row];states[row]=0;
 // A row produces four corrections in canonical node order; publication is separate.
 for(unsigned j=0;j<4;++j)corrections[4*row+j]=make_float4(0,0,0,0);
 if(at>=capacity||nodes>65536){atomicAdd(errors,1);states[row]=6;return;}
 FH entry=history[at];states[row]=entry.state;
 if(!entry.key||entry.epoch!=epoch||entry.seen!=epoch){atomicAdd(errors,1);states[row]=6;return;}
 if(entry.state!=1&&entry.state!=5)return;
 if(!isfinite(separation)||separation<0||!isfinite(h)||h<=0){atomicAdd(errors,1);states[row]=6;return;}
 unsigned ids[4]={unsigned(entry.key>>48),unsigned((entry.key>>32)&65535),unsigned((entry.key>>16)&65535),unsigned(entry.key&65535)};
 float4 cur[4];for(int j=0;j<4;++j){if(ids[j]>=nodes){atomicAdd(errors,1);states[row]=6;return;}cur[j]=positions[ids[j]];if(!finite4(cur[j])){atomicAdd(errors,1);states[row]=6;return;}}
 SV n;double gap,s,t;if(!geometry(cur,n,gap,s,t)){atomicAdd(errors,1);states[row]=6;return;}
 n=mul(n,entry.side);gap*=entry.side;double w[4]={1-s,s,-(1-t),-t},den=0;
 for(int j=0;j<4;++j)den+=cur[j].w*w[j]*w[j];if(!(den>1e-20)){atomicAdd(errors,1);states[row]=6;return;}
 double lambda=fmax(0.,separation-gap)/den;
 for(int j=0;j<4;++j){SV d=mul(n,lambda*cur[j].w*w[j]);corrections[4*row+j]=make_float4(float(d.x),float(d.y),float(d.z),0);}
}
