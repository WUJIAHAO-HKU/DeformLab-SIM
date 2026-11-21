// Complete native query's support: its >1e-3 branches already counted the
// other positive weights. Preserve its activity decisions and rigid counts.
extern "C" __global__ void complete_contact_support(char* softbodies,const uint4* info,
    const char* constraints,const unsigned* numContacts,unsigned* rejected){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=*numContacts)return;
 uint4 ci=info[i];if(!(ci.w&0x80000000u))return;
 char* body=softbodies+size_t(ci.z>>20)*432;unsigned tet=ci.z&0xfffffu;
 if(tet>=field<unsigned>(body,376)){atomicAdd(rejected,1u);return;}
 uint4 ids=field<uint4*>(body,80)[tet];
 float4 bc=*reinterpret_cast<const float4*>(constraints+size_t(i/32)*3200+2560+16*(i%32));
 auto d=field<float4*>(body,272);auto pos=field<float4*>(body,128);
 for(int k=0;k<4;++k){float w=(&bc.x)[k];unsigned node=(&ids.x)[k];
  if(!isfinite(w)||w<0 || node>=field<unsigned>(body,372)){atomicAdd(rejected,1u);continue;}
  if(w>0.f && w<=1e-3f && pos[node].w!=0.f)atomicAdd(&d[node].w,1.f);
 }
}
