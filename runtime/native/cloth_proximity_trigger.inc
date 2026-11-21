// Read-only pre-integration trigger on six fixed root-node identities.
struct CTV {double x,y,z;};
static CTV ct_sub(CTV a,CTV b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static CTV ct_add(CTV a,CTV b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static CTV ct_mul(CTV a,double v){return {a.x*v,a.y*v,a.z*v};}
static double ct_dot(CTV a,CTV b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static CTV ct_cross(CTV a,CTV b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static double ct_clamp(double x){return std::max(0.,std::min(1.,x));}
static double ct_ps(CTV p,CTV a,CTV b){CTV e=ct_sub(b,a),v=ct_sub(p,a);double ee=ct_dot(e,e),t=ee>0?ct_clamp(ct_dot(v,e)/ee):0;CTV d=ct_sub(v,ct_mul(e,t));return ct_dot(d,d);}
static double ct_pt(CTV p,const CTV* t){
 CTV e=ct_sub(t[1],t[0]),f=ct_sub(t[2],t[0]),r=ct_sub(p,t[0]),n=ct_cross(e,f);double nn=ct_dot(n,n);
 double a=ct_dot(ct_cross(r,f),n)/nn,b=ct_dot(ct_cross(e,r),n)/nn;
 double best=std::min(ct_ps(p,t[0],t[1]),std::min(ct_ps(p,t[1],t[2]),ct_ps(p,t[2],t[0])));
 if(a>=0&&b>=0&&a+b<=1){double d=ct_dot(r,n);best=std::min(best,d*d/nn);}return best;
}
static double ct_ss(CTV a,CTV b,CTV c,CTV d){
 CTV u=ct_sub(b,a),v=ct_sub(d,c),w=ct_sub(a,c),n=ct_cross(u,v);double nn=ct_dot(n,n);
 double best=std::min(std::min(ct_ps(a,c,d),ct_ps(b,c,d)),std::min(ct_ps(c,a,b),ct_ps(d,a,b)));
 if(nn>1e-30*ct_dot(u,u)*ct_dot(v,v)){
  double s=ct_dot(ct_cross(v,w),n)/nn,t=ct_dot(ct_cross(u,w),n)/nn;
  if(s>=0&&s<=1&&t>=0&&t<=1){CTV r=ct_sub(ct_add(w,ct_mul(u,s)),ct_mul(v,t));best=std::min(best,ct_dot(r,r));}
 }return best;
}
static bool ct_segment_pierces(CTV a,CTV b,const CTV* t){
 CTV e=ct_sub(t[1],t[0]),f=ct_sub(t[2],t[0]),n=ct_cross(e,f);double nn=ct_dot(n,n);
 double da=ct_dot(ct_sub(a,t[0]),n),db=ct_dot(ct_sub(b,t[0]),n);
 if(!((da>0&&db<0)||(da<0&&db>0)))return false;
 CTV r=ct_sub(ct_add(a,ct_mul(ct_sub(b,a),da/(da-db))),t[0]);double u=ct_dot(ct_cross(r,f),n)/nn,v=ct_dot(ct_cross(e,r),n)/nn;
 return u>=0&&v>=0&&u+v<=1;
}
static bool ct_triangle_distance(const float* xyz,double& value){
 CTV p[6];for(unsigned i=0;i<6;++i){p[i]={xyz[3*i],xyz[3*i+1],xyz[3*i+2]};if(!std::isfinite(p[i].x)||!std::isfinite(p[i].y)||!std::isfinite(p[i].z))return false;}
 for(unsigned k:{0u,3u}){CTV n=ct_cross(ct_sub(p[k+1],p[k]),ct_sub(p[k+2],p[k]));if(ct_dot(n,n)<=1e-28)return false;}
 double best=1e300;
 for(unsigned i=0;i<3;++i){
  if(ct_segment_pierces(p[i],p[(i+1)%3],p+3)||ct_segment_pierces(p[3+i],p[3+(i+1)%3],p)){value=0;return true;}
  best=std::min(best,std::min(ct_pt(p[i],p+3),ct_pt(p[3+i],p)));
  for(unsigned j=0;j<3;++j)best=std::min(best,ct_ss(p[i],p[(i+1)%3],p[3+j],p[3+(j+1)%3]));
 }
 value=std::sqrt(std::max(0.,best));return std::isfinite(value);
}
static bool ca_enabled=false,ca_fired=false;static FILE* ca_log=nullptr;
static unsigned ca_nodes[6]{},ca_span=0,ca_horizon=0,ca_samples=0,ca_first=0,ca_last=0,ca_errors=0;
static double ca_threshold=0,ca_minimum=1e300,ca_distance=1e300;
static void ca_reset(){ca_enabled=ca_fired=false;ca_span=ca_horizon=ca_samples=ca_first=ca_last=ca_errors=0;ca_threshold=0;ca_minimum=ca_distance=1e300;}
static bool ca_fail(){++ca_errors;return cs_fail();}
static bool ca_probe(){
 if(!ca_enabled)return true;
 if(cs_step>ca_horizon||!ca_log)return ca_fail();
 float xyz[18];for(unsigned i=0;i<6;++i){float row[4];if(ca_nodes[i]>=cs_n||!cs_read(row,cs_ptrs[0]+16ull*ca_nodes[i],16))return ca_fail();for(unsigned k=0;k<3;++k)xyz[3*i+k]=row[k];}
 if(!ct_triangle_distance(xyz,ca_distance))return ca_fail();++ca_samples;ca_minimum=std::min(ca_minimum,ca_distance);
 bool fired_now=false;
 if(!ca_fired&&ca_distance<=ca_threshold){ca_fired=true;fired_now=true;ca_first=cs_step;ca_last=std::min(ca_horizon,cs_step+ca_span-1);cs_first=ca_first;cs_last=ca_last;}
 if(std::fprintf(ca_log,"{\"step\":%u,\"previous_end_step\":%u,\"distance_m\":%.17g,\"triggered_now\":%s,\"positions_m\":[",cs_step,cs_step-1,ca_distance,fired_now?"true":"false")<0)return ca_fail();
 for(unsigned i=0;i<18;++i)if(std::fprintf(ca_log,"%s%.9g",i?",":"",xyz[i])<0)return ca_fail();
 if(std::fprintf(ca_log,"]}\n")<0)return ca_fail();return true;
}
static bool ca_close(){if(ca_log){if(std::fclose(ca_log)){ca_log=nullptr;return ca_fail();}ca_log=nullptr;}return true;}
