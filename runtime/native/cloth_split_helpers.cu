// Appended to the guarded contact module by the pinned build script.
extern "C" __global__ void cloth_split_prepare(
 unsigned char* table,unsigned char* normal_table,unsigned char* friction_table,
 float4* saved_positions,float4* saved_degree,unsigned n,const unsigned* ids,
 unsigned long long expected_p,unsigned long long expected_v,unsigned long long expected_pending){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i==0){
  if(*ids!=0||field<unsigned>(table,316)!=n||field<unsigned long long>(table,16)!=expected_p||
     field<unsigned long long>(table,8)!=expected_v||field<unsigned long long>(table,248)!=expected_pending)reject();
 }
 if(i<50){reinterpret_cast<unsigned long long*>(normal_table)[i]=reinterpret_cast<unsigned long long*>(table)[i];
          reinterpret_cast<unsigned long long*>(friction_table)[i]=reinterpret_cast<unsigned long long*>(table)[i];}
 if(i<n){
  auto p=reinterpret_cast<const float4*>(expected_p);auto pending=reinterpret_cast<const float4*>(expected_pending);
  saved_positions[i]=p[i];saved_degree[i]=pending[i];
  if(pending[i].x!=0||pending[i].y!=0||pending[i].z!=0)reject();
 }
}
extern "C" __global__ void cloth_split_bind_frozen(unsigned char* table,float4* saved_positions){
 if(threadIdx.x==0&&blockIdx.x==0)*reinterpret_cast<float4**>(table+16)=saved_positions;
}
extern "C" __global__ void cloth_split_restore_degree(float4* pending,const float4* saved,unsigned n){
 if(cloth_split_counts[6])return;
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n)pending[i]=make_float4(0,0,0,saved[i].w);
}
