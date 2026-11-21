// First-rejection diagnostics only. No solver-state writes.
extern "C" { __device__ __align__(16) unsigned cloth_history_first_failure[128]; __device__ unsigned cloth_history_first_decode[16]; }
__device__ unsigned long long fh_diagnostic_previous=0,fh_diagnostic_frame=0;
__device__ void fh_record_failure(unsigned reason,unsigned phase,unsigned row,uint4 record,const unsigned char* table,const FH* history,const unsigned* slots,unsigned capacity,unsigned epoch,float h,bool edge){
 if(atomicCAS(cloth_history_first_failure,0u,1u)!=0)return;
 unsigned* dst=cloth_history_first_failure;dst[1]=reason;dst[2]=phase;dst[3]=row;dst[4]=epoch;dst[5]=field<unsigned>(table,316);dst[6]=record.x;dst[7]=record.y;dst[8]=record.z;dst[9]=record.w;dst[10]=~0u;
 if(history&&slots){unsigned at=slots[row];dst[10]=at;if(at<capacity){FH entry=history[at];dst[11]=unsigned(entry.side);dst[12]=entry.state;dst[13]=entry.epoch;dst[14]=entry.seen;dst[15]=unsigned(entry.key);dst[16]=unsigned(entry.key>>32);}}
 unsigned n=dst[5],nt=field<unsigned>(table,320);dst[17]=nt;for(unsigned i=0;i<4;++i)dst[18+i]=~0u;
 const uint4* tri=field<const uint4*>(table,72);unsigned a=record.x&0xfffff,b=record.z&0xfffff,ea=record.w&16383,eb=(record.w>>14)&32767;
 if(!(record.x>>20)&&!(record.z>>20)&&b<nt&&((edge&&a<nt&&ea<3&&eb<3)||(!edge&&a<n))){
  uint4 tb=tri[b];unsigned vb[3]={tb.x,tb.y,tb.z};
  if(edge){uint4 ta=tri[a];unsigned va[3]={ta.x,ta.y,ta.z};dst[18]=va[ea];dst[19]=va[(ea+1)%3];dst[20]=vb[eb];dst[21]=vb[(eb+1)%3];}
  else{dst[18]=a;dst[19]=tb.x;dst[20]=tb.y;dst[21]=tb.z;}
  const float4* positions=field<const float4*>(table,16);const float4* old=reinterpret_cast<const float4*>(fh_diagnostic_previous);const float4* native_old=field<const float4*>(table,32);
  for(unsigned j=0;j<4;++j)if(dst[18+j]<n){reinterpret_cast<float4*>(dst+24)[j]=positions[dst[18+j]];if(old)reinterpret_cast<float4*>(dst+40)[j]=old[dst[18+j]];if(native_old)reinterpret_cast<float4*>(dst+56)[j]=native_old[dst[18+j]];}
 }
 dst[72]=unsigned(fh_diagnostic_frame);dst[73]=unsigned(fh_diagnostic_frame>>32);dst[74]=__float_as_uint(field<float>(table,384));dst[75]=__float_as_uint(h);dst[76]=edge?1:0;__threadfence();atomicExch(dst,2u);
}
__device__ void fh_record_decode(unsigned reason,unsigned row,uint4 record,unsigned count,unsigned epoch){
 unsigned* dst=cloth_history_first_decode;if(atomicCAS(dst,0u,1u)!=0)return;dst[1]=reason;dst[2]=row;dst[3]=count;dst[4]=epoch;dst[5]=record.x;dst[6]=record.y;dst[7]=record.z;dst[8]=record.w;dst[9]=unsigned(fh_diagnostic_frame);__threadfence();atomicExch(dst,2u);
}
