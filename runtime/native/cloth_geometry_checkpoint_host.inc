
// Read-only checkpoint of the existing pre-EE ring at a caller-verified frame.
// Preserve the ordinary end-of-run stream and counters; never mutate GPU state.
extern "C" int cloth_history_geometry_checkpoint(
 const char* path,unsigned long long expected_frame,unsigned long long* values){
 std::lock_guard<std::mutex> guard(mutex);
 if(!path||!*path||!values||!cf_enabled||!cf_context||!fh_enabled||!fhr_enabled||
    !fhr_file||!fhr_metadata||!fh_epoch||!expected_frame||
    expected_frame!=fh_frames||fh_geometry_checkpoint_used)return -1;
 CUresult rc=cuCtxPushCurrent(cf_context);if(rc!=CUDA_SUCCESS)return int(rc);
 rc=cuCtxSynchronize();
 if(rc!=CUDA_SUCCESS){CUcontext popped;cuCtxPopCurrent(&popped);return int(rc);}
 FILE* checkpoint=std::fopen(path,"wx");
 if(!checkpoint){CUcontext popped;cuCtxPopCurrent(&popped);return -1;}
 FILE* original=fhr_file;unsigned long long previous[9];std::copy(fhr_last,fhr_last+9,previous);
 fhr_file=checkpoint;bool ok=fhr_flush(fh_epoch);
 unsigned long long captured[9];std::copy(fhr_last,fhr_last+9,captured);
 fhr_file=original;std::copy(previous,previous+9,fhr_last);
 CUcontext popped=nullptr;rc=cuCtxPopCurrent(&popped);
 if(!ok||rc!=CUDA_SUCCESS||popped!=cf_context)return -1;
 values[0]=fh_epoch;values[1]=fh_frames;std::copy(captured,captured+9,values+2);
 fh_geometry_checkpoint_used=true;return 0;
}
