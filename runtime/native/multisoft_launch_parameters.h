// CPU-side launch-argument capture. No CUDA calls or device writes.
#pragma once
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
struct MsLaunchScalar { unsigned index, component, bits; float value; bool byte; };
inline bool ms_launch_parameters(FILE* out,const char* name,void** params){
 if(!out||!name)return false;
 MsLaunchScalar scalars[8];unsigned size=0;bool valid=true,known=true;
 auto scalar=[&](unsigned index,unsigned count=1,bool byte=false){
  if(!params||!params[index]||size+count>8){valid=false;return;}
  for(unsigned c=0;c<count;++c){
   MsLaunchScalar s{index,c,0,0,byte};
   if(byte){std::uint8_t value;std::memcpy(&value,params[index],1);s.bits=value;s.value=value;}
   else{const auto* p=static_cast<const unsigned char*>(params[index])+c*4;std::memcpy(&s.value,p,4);std::memcpy(&s.bits,p,4);if(!std::isfinite(s.value))valid=false;}
   scalars[size++]=s;
  }
 };
 if(!std::strcmp(name,"sb_gm_applyExternalDeltasLaunch")||!std::strcmp(name,"sb_gm_cp_averageVertsLaunch")){scalar(2);}
 else if(!std::strcmp(name,"sb_gm_stepSoftbodyLaunch")){scalar(2);scalar(3,3);scalar(4,1,true);}
 else if(!std::strcmp(name,"sb_gm_finalizeVelocitiesLaunch")){scalar(2);scalar(3);scalar(4);scalar(5,1,true);}
 else if(!std::strcmp(name,"sb_gm_preIntegrateLaunch")){scalar(2,3);scalar(3);scalar(4,1,true);scalar(6,1,true);}
 else known=false;
 if(!valid)return false;
 if(!known)return std::fprintf(out,"null")>=0;
 if(std::fprintf(out,"{\"schema_version\":1,\"scalars\":[")<0)return false;
 for(unsigned i=0;i<size;++i){const auto& s=scalars[i];
  if(std::fprintf(out,"%s{\"index\":%u,\"component\":%u,\"type\":\"%s\",\"bits\":%u,\"value\":",i?",":"",s.index,s.component,s.byte?"u8":"f32",s.bits)<0)return false;
  if(s.value==0&&std::signbit(s.value)){if(std::fprintf(out,"-0.0")<0)return false;}
  else if(std::fprintf(out,"%.17g",double(s.value))<0)return false;
  if(std::fprintf(out,"}")<0)return false;
 }
 return std::fprintf(out,"]}")>=0&&!std::ferror(out);
}
