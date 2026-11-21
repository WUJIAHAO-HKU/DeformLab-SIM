// Explicit opt-in batching of the unchanged seven owned contact kernels.
// Included after cf_allocate. Context, tables and scratch remain lifecycle-bound.
static bool cg_enabled=false;
static unsigned long long cg_counts[5]{}; // builds, reuses, launches, destroys, errors
struct ClothGraph {
 CUgraph graph=nullptr; CUgraphExec exec=nullptr;
 CUdeviceptr rows=0,count=0; unsigned gx=0,bx=0; float h=0;
};
static ClothGraph cg_cache[2];
static bool cg_destroy(ClothGraph& c){
 if(c.exec){if(cuGraphExecDestroy(c.exec)!=CUDA_SUCCESS){++cg_counts[4];return false;}c.exec=nullptr;++cg_counts[3];}
 if(c.graph){if(cuGraphDestroy(c.graph)!=CUDA_SUCCESS){++cg_counts[4];return false;}c.graph=nullptr;}
 return true;
}
static CUresult cg_dispatch(Launch original,int kind,unsigned gx,unsigned bx,CUstream stream,void** params){
 CUdeviceptr rows=*static_cast<CUdeviceptr*>(params[1]),count=*static_cast<CUdeviceptr*>(params[2]);
 auto& c=cg_cache[kind];
 if(!gx||!rows||!count){++cg_counts[4];return cf_fail();}
 bool match=c.exec&&c.rows==rows&&c.count==count&&c.gx==gx&&c.bx==bx&&c.h==cf_h;
 if(!match){
  if(c.exec&&cuCtxSynchronize()!=CUDA_SUCCESS){++cg_counts[4];return cf_fail();}
  if(!cg_destroy(c))return cf_fail();
  if(cuGraphCreate(&c.graph,0)!=CUDA_SUCCESS){++cg_counts[4];return cf_fail();}
  CUgraphNode last=nullptr;
  auto add=[&](CUfunction f,unsigned grid,unsigned block,void** p){
   CUDA_KERNEL_NODE_PARAMS k{};k.func=f;k.gridDimX=grid;k.gridDimY=k.gridDimZ=1;
   k.blockDimX=block;k.blockDimY=k.blockDimZ=1;k.kernelParams=p;
   CUgraphNode node=nullptr;
   if(cuGraphAddKernelNode(&node,c.graph,last?&last:nullptr,last?1:0,&k)!=CUDA_SUCCESS)return false;
   last=node;return true;
  };
  void* prepare[]={&cf_table,&cf_normal_table,&cf_friction_table,&cf_positions,&cf_degree,&cf_n,&cf_ids,&cf_p,&cf_v,&cf_delta};
  void* frozen[]={&cf_friction_table,&cf_positions};
  void* normal[]={&cf_normal_table,&rows,&count,&cf_h};
  void* publish[]={&cf_table,&cf_ids,&cf_h};
  void* restore[]={&cf_delta,&cf_degree,&cf_n};
  void* friction[]={&cf_friction_table,&rows,&count,&cf_h};
  if(!add(cf_prepare,(cf_n+255)/256,256,prepare)||!add(cf_bind_frozen,1,1,frozen)||
     !add(cf_query[kind],gx,bx,normal)||!add(cf_normal[kind],gx,bx,normal)||
     !add(cf_publish,(cf_n+63)/64,64,publish)||!add(cf_restore,(cf_n+255)/256,256,restore)||
     !add(cf_friction[kind],gx,bx,friction)||cuGraphInstantiate(&c.exec,c.graph,0)!=CUDA_SUCCESS){
    ++cg_counts[4];cg_destroy(c);return cf_fail();
  }
  c.rows=rows;c.count=count;c.gx=gx;c.bx=bx;c.h=cf_h;++cg_counts[0];
 }else ++cg_counts[1];
 // Only these owned graphs bypass the unsupported-external-graph census.
 // Retain the intercepted call's legacy/per-thread default-stream semantics.
 using GraphLaunch=CUresult(CUDAAPI*)(CUgraphExec,CUstream);
 static auto ptsz=reinterpret_cast<Launch>(lookup()(RTLD_NEXT,"cuLaunchKernel_ptsz"));
 static auto normal_graph=reinterpret_cast<GraphLaunch>(lookup()(RTLD_NEXT,"cuGraphLaunch"));
 static auto ptsz_graph=reinterpret_cast<GraphLaunch>(lookup()(RTLD_NEXT,"cuGraphLaunch_ptsz"));
 auto fn=original==ptsz?ptsz_graph:normal_graph;
 if(!fn||fn(c.exec,stream)!=CUDA_SUCCESS){++cg_counts[4];return cf_fail();}
 ++cg_counts[2];return CUDA_SUCCESS;
}
