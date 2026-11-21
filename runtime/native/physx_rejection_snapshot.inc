// Read-only, explicit outer-boundary diagnostics. No solver flag is reset.
// Caller must own a quiescent simulation boundary; not for concurrent stepping.
extern "C" int observer_rejection_snapshot(unsigned long long* out,unsigned count){
 if(!out || count!=8)return -2;
 std::lock_guard<std::mutex> guard(lock);
 if(!enabled || !context || !rejected)return -1;
 unsigned device_bad=0;
 CUresult result=cuCtxPushCurrent(context);
 if(result!=CUDA_SUCCESS)return int(result);
 result=cuCtxSynchronize();
 if(result==CUDA_SUCCESS)result=cuMemcpyDtoH(&device_bad,rejected,sizeof(device_bad));
 CUcontext previous=nullptr;CUresult popped=cuCtxPopCurrent(&previous);
 if(result!=CUDA_SUCCESS)return int(result);
 if(popped!=CUDA_SUCCESS)return int(popped);
 unsigned long long snapshot[8]={errors.load(),device_bad,pc_rejections,
   static_cast<unsigned long long>(cp_frame),dc_drives,dc_contacts,dc_applies,
   static_cast<unsigned long long>(dc_pending || dc_have_drive)};
 std::memcpy(out,snapshot,sizeof(snapshot));return 0;
}
