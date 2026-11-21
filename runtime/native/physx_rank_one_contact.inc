// Explicit experimental device-only coupled contact adapter, fixed coupon scope.
namespace {
CUmodule nr_module=nullptr,nr_transport_module=nullptr;
CUfunction nr_decode=nullptr,nr_group=nullptr,nr_compact=nullptr,nr_solve=nullptr,nr_publish=nullptr,nr_transport=nullptr;
CUdeviceptr nr_rows=0,nr_groups=0,nr_inputs=0,nr_map=0,nr_solved=0,nr_target=0;
unsigned long long nr_calls=0;
void nr_free(){
 for(auto p:{nr_rows,nr_groups,nr_inputs,nr_map,nr_solved,nr_target})if(p)cuMemFree(p);
 nr_rows=nr_groups=nr_inputs=nr_map=nr_solved=nr_target=0;
 if(nr_module)cuModuleUnload(nr_module);if(nr_transport_module)cuModuleUnload(nr_transport_module);
 nr_module=nr_transport_module=nullptr;nr_decode=nr_group=nr_compact=nr_solve=nr_publish=nr_transport=nullptr;nr_calls=0;
}
CUresult nr_launch(Launch original,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,unsigned shared,CUstream stream,void** params){
 if(!sb || !rejected || !terminal_transport || !coupled_damping || !tp_module || gy!=1 || gz!=1 || by!=1 || bz!=1)
  return CUDA_ERROR_NOT_SUPPORTED;
 float h=*static_cast<float*>(params[11]);
 if(!std::isfinite(h) || h<=0.f || std::fabs(h-tp_h)>2e-6f*tp_h)return CUDA_ERROR_NOT_SUPPORTED;
 CUstreamCaptureStatus capture;CUresult c=cuStreamIsCapturing(stream,&capture);
 if(c!=CUDA_SUCCESS || capture!=CU_STREAM_CAPTURE_STATUS_NONE)return CUDA_ERROR_NOT_SUPPORTED;
 unsigned n=unsigned(nodes);int faces=2;
 void* decode[]={params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[9],params[11],&sb,&n,&nr_rows,&rejected};
 c=real_launch()(nr_decode,8,1,1,128,1,1,0,stream,decode,nullptr);
 void* group[]={&nr_rows,params[3],&positions,&n,&nr_groups,&rejected};
 if(c==CUDA_SUCCESS)c=real_launch()(nr_group,(n+127)/128,1,1,128,1,1,0,stream,group,nullptr);
 void* compact[]={&nr_groups,&n,params[11],&nr_inputs,&nr_map,&rejected};
 if(c==CUDA_SUCCESS)c=real_launch()(nr_compact,1,1,1,1,1,1,0,stream,compact,nullptr);
 void* solve[]={&nr_inputs,&nr_solved,&faces};
 if(c==CUDA_SUCCESS)c=real_launch()(nr_solve,2,1,1,128,1,1,0,stream,solve,nullptr);
 void* publish[]={&nr_rows,params[3],&n,&nr_groups,&nr_map,&nr_solved,&nr_target,&rejected};
 if(c==CUDA_SUCCESS)c=real_launch()(nr_publish,8,1,1,128,1,1,0,stream,publish,nullptr);
 void* mapped[16];for(int i=0;i<14;++i)mapped[i]=params[i];mapped[14]=&nr_target;mapped[15]=&rejected;
 if(c==CUDA_SUCCESS)c=original(nr_transport,gx,gy,gz,bx,by,bz,shared,stream,mapped,nullptr);
 if(c==CUDA_SUCCESS)++nr_calls;return c;
}
}
extern "C" int observer_rank_one_contact_start(const char* adapter,const char* transport){
 if(!adapter||!transport||!enabled||mode!=6||normal_policy!=2||!terminal_transport||!coupled_damping||!tp_module||
    nr_module||nr_transport_module||nr_rows||damping_calls||nodes==0||nodes>4096||!rejected)return -1;
 CUresult c=cuCtxPushCurrent(context);if(c!=CUDA_SUCCESS)return int(c);
 c=cuModuleLoad(&nr_module,adapter);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&nr_decode,nr_module,"decode_rank_one_rows");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&nr_group,nr_module,"group_rank_one_nodes");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&nr_compact,nr_module,"compact_rank_one_groups");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&nr_solve,nr_module,"solve_rank_one");
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&nr_publish,nr_module,"publish_rank_one_targets");
 if(c==CUDA_SUCCESS)c=cuModuleLoad(&nr_transport_module,transport);
 if(c==CUDA_SUCCESS)c=cuModuleGetFunction(&nr_transport,nr_transport_module,"sb_solveRigidSoftCollisionLaunchTGS");
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&nr_rows,1024*6*sizeof(double));
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&nr_groups,2*nodes*4*sizeof(double));
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&nr_inputs,2*515*sizeof(double));
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&nr_map,2*nodes*sizeof(int));
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&nr_solved,2*130*sizeof(double));
 if(c==CUDA_SUCCESS)c=cuMemAlloc(&nr_target,1024*sizeof(float));
 if(c==CUDA_SUCCESS)nr_calls=0;else nr_free();
 CUcontext previous;cuCtxPopCurrent(&previous);return int(c);
}
extern "C" unsigned long long observer_rank_one_contact_count(){return nr_calls;}
