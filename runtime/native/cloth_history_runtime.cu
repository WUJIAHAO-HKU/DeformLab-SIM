// Appended after the verified history/split source. Independent frame history.
extern "C" __global__ void cloth_history_capture_previous(unsigned char* table,float4* previous,unsigned nodes,unsigned long long expected_positions,unsigned long long expected_triangles,unsigned triangles,unsigned* errors,unsigned long long* counters){
 unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
 if(field<unsigned>(table,316)!=nodes||field<unsigned>(table,320)!=triangles||field<unsigned long long>(table,16)!=expected_positions||field<unsigned long long>(table,72)!=expected_triangles){if(i==0){atomicAdd(errors,1);atomicAdd(cloth_split_counts+6,1ull);}return;}
 if(i==0)atomicAdd(counters+6,1ull);
 if(i<nodes)previous[i]=reinterpret_cast<const float4*>(expected_positions)[i];
}
extern "C" __global__ void cloth_history_guard(const unsigned* errors,const unsigned long long* counters){
 if(blockIdx.x||threadIdx.x)return;
 if(*errors||counters[4])atomicAdd(cloth_split_counts+6,1ull);
}
