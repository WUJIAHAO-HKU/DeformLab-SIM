// Matched-binary read-only preparation capture. Separate stream and lifecycle.
// Does not alter CUDA arguments, simulation buffers or contact weights.
// Synchronous readback adds observation overhead; never a performance sample.
namespace {
FILE* gp_index=nullptr;FILE* gp_blob=nullptr;
bool gp_pending=false;unsigned long long gp_seq=0,gp_records=0;
int gp_step=-1;
bool gp_copy(void* out,CUdeviceptr p,size_t size){
 if(!size)return true;
 CUdeviceptr base=0;size_t available=0;
 if(!p||cuMemGetAddressRange(&base,&available,p)!=CUDA_SUCCESS||p<base||
    p-base>available||size>available-size_t(p-base))return false;
 return cuMemcpyDtoH(out,p,size)==CUDA_SUCCESS;
}
bool gp_start(const char* path){
 if(!path||gp_index||gp_blob||gp_pending)return false;
 gp_records=0;gp_seq=0;gp_step=-1;
 gp_index=std::fopen((std::string(path)+"-preparation.jsonl").c_str(),"wx");
 gp_blob=std::fopen((std::string(path)+"-preparation.bin").c_str(),"wx");
 return gp_index&&gp_blob;
}
void gp_stop(){
 if(gp_pending)++errors;
 if(gp_index){if(std::fclose(gp_index))++errors;gp_index=nullptr;}
 if(gp_blob){if(std::fclose(gp_blob))++errors;gp_blob=nullptr;}
 gp_pending=false;gp_seq=0;gp_step=-1;
}
bool gp_selected(const char* name,int step){
 return gp_index&&enabled&&step>=0&&!std::strcmp(name,"sb_rigidContactPrepareLaunch");
}
bool gp_snapshot(void** params,bool before,unsigned long long seq,int step){
 auto fail=[](){++errors;return false;};
 if(!gp_index||!gp_blob||!enabled||!params||step<0)return fail();
 if((before&&gp_pending)||(!before&&(!gp_pending||gp_seq!=seq||gp_step!=step)))return fail();
 CUcontext current=nullptr;
 if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS||current!=context||cuCtxSynchronize()!=CUDA_SUCCESS)return fail();
 for(int i:{1,2,3,4,5,6,7,8,11,12})if(!params[i])return fail();
 auto arg=[&](unsigned i){return *static_cast<CUdeviceptr*>(params[i]);};
 unsigned count=0;unsigned char prep[320],shared_desc[96],preprep[760];
 if(!gp_copy(&count,arg(5),4)||count>16384||!gp_copy(prep,arg(8),sizeof(prep))||
    !gp_copy(shared_desc,arg(11),sizeof(shared_desc))||!gp_copy(preprep,arg(7),sizeof(preprep)))return fail();
 unsigned total=field<unsigned>(prep,248);
 if(total>4096)return fail();
 std::vector<unsigned> info(size_t(count)*4),identity(size_t(count)*4);
 std::vector<float> point(size_t(count)*4),normal(size_t(count)*4),weights(size_t(count)*4),
                    origin(size_t(count)*3),sqrt_inertia(size_t(count)*9);
 std::vector<unsigned char> constraints(size_t((count+31)/32)*3200);
 if(!gp_copy(info.data(),arg(4),info.size()*4)||!gp_copy(point.data(),arg(1),point.size()*4)||
    !gp_copy(normal.data(),arg(2),normal.size()*4)||!gp_copy(weights.data(),arg(3),weights.size()*4)||
    !gp_copy(constraints.data(),arg(6),constraints.size()))return fail();
 for(unsigned i=0;i<count;++i){
  unsigned raw=info[4*i],flags=info[4*i+1];bool is_static=raw==~0u,is_art=(flags&1u)!=0;
  if(is_static&&is_art)return fail();
  if((info[4*i+2]&0xfffffu)==0xfffffu){
   for(unsigned j=0;j<4;++j)identity[4*i+j]=~0u;
   continue; // Native preparation also skips this invalid tetrahedron marker.
  }
  if(!is_static&&raw>1048575u)return fail();
  float xyz[3];unsigned slot=0,kind=0,link=~0u;
  if(is_art){
   unsigned char body[224],arti[496],transform[28];unsigned remap=0;
   CUdeviceptr bodies=field<CUdeviceptr>(shared_desc,32),articulations=field<CUdeviceptr>(shared_desc,40);
   if(!bodies||!articulations||!gp_copy(body,bodies+size_t(raw)*224,sizeof(body)))return fail();
   remap=field<unsigned>(body,176);link=flags>>1;
   if(remap>=4096||!gp_copy(arti,articulations+size_t(remap)*496,sizeof(arti)))return fail();
   unsigned links=field<unsigned short>(arti,16);
   if(!links||links>128||link>=links)return fail();
   CUdeviceptr frames=field<CUdeviceptr>(arti,176);
   if(!frames||!gp_copy(transform,frames+size_t(link)*28,sizeof(transform)))return fail();
   for(unsigned j=0;j<3;++j)xyz[j]=field<float>(transform,16+4*j);
   kind=3;slot=remap;
   for(unsigned j=0;j<3;++j)sqrt_inertia[9*i+3*j+j]=1.f;
  }else{
   CUdeviceptr ids=field<CUdeviceptr>(preprep,272);
   if(!is_static&&(!ids||!gp_copy(&slot,ids+size_t(raw)*4,4)))return fail();
   // The native static branch selects slot zero independently of body count.
   if(!is_static&&slot>=total)return fail();
   unsigned char transform[32],body[96],txi[64];
   CUdeviceptr frames=field<CUdeviceptr>(prep,168),body_data=field<CUdeviceptr>(prep,64),tx=field<CUdeviceptr>(prep,72);
   if(!frames||!body_data||!tx||!gp_copy(transform,frames+size_t(slot)*32,sizeof(transform))||
      !gp_copy(body,body_data+size_t(slot)*96,sizeof(body))||!gp_copy(txi,tx+size_t(slot)*64,sizeof(txi)))return fail();
   float inv_mass=field<float>(body,28);
   if(!std::isfinite(inv_mass)||inv_mass<0)return fail();
   kind=is_static?0:inv_mass==0.f?2:1;
   for(unsigned j=0;j<3;++j)xyz[j]=field<float>(transform,16+4*j);
   for(unsigned j=0;j<9;++j)sqrt_inertia[9*i+j]=field<float>(txi,28+4*j);
  }
  for(unsigned j=0;j<3;++j){
   if(!std::isfinite(xyz[j])||!std::isfinite(point[4*i+j])||!std::isfinite(normal[4*i+j]))return fail();
   origin[3*i+j]=xyz[j];
  }
  for(unsigned j=0;j<9;++j)if(!std::isfinite(sqrt_inertia[9*i+j]))return fail();
  identity[4*i]=kind;identity[4*i+1]=slot;identity[4*i+2]=link;identity[4*i+3]=raw;
 }
 std::fprintf(gp_index,"{\"schema\":1,\"seq\":%llu,\"step\":%d,\"phase\":\"%s\",\"contacts\":%u,\"tgs\":%u,\"sections\":{",seq,step,before?"before":"after",count,unsigned(*static_cast<unsigned char*>(params[12])));
 bool first=true,ok=true;
 auto host=[&](const char* name,const void* bytes,size_t size){
  if(!ok)return;long offset=std::ftell(gp_blob);
  if(offset<0||(size&&std::fwrite(bytes,1,size,gp_blob)!=size)){ok=false;return;}
  if(!first)std::fprintf(gp_index,",");first=false;
  std::fprintf(gp_index,"\"%s\":[%ld,%zu]",name,offset,size);
 };
 host("native_info",info.data(),info.size()*4);host("actor_identity",identity.data(),identity.size()*4);
 host("point_rest",point.data(),point.size()*4);host("normal_pen",normal.data(),normal.size()*4);
 host("barycentric",weights.data(),weights.size()*4);host("actor_origin",origin.data(),origin.size()*4);
 host("sqrt_inv_inertia_column_major",sqrt_inertia.data(),sqrt_inertia.size()*4);
 host("constraint_blocks",constraints.data(),constraints.size());
 std::fprintf(gp_index,"}}\n");
 if(!ok||std::ferror(gp_index)||std::fflush(gp_index)||std::fflush(gp_blob))return fail();
 gp_pending=before;gp_seq=seq;gp_step=step;++gp_records;return true;
}
}
