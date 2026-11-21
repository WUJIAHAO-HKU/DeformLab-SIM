// Host-only sparse contact projection using the shared normal primitive.
#include "contact_normal_history.h"
#include <algorithm>
#include <cmath>
#include <vector>
extern "C" int terminal_normal(unsigned nodes,unsigned rows,const unsigned* ids,const double* weights,
 const double* normals,const double* inverse,const double* freev,unsigned budget,double tolerance,
 double* velocity,double* impulses,unsigned* iterations){
 if(!nodes||!rows||!ids||!weights||!normals||!inverse||!freev||!velocity||!impulses||!iterations||!budget||!(tolerance>0)||!std::isfinite(tolerance))return -1;
 std::vector<double> mobility(rows);
 for(unsigned i=0;i<nodes;++i){if(!std::isfinite(inverse[i])||inverse[i]<0)return -2;for(unsigned d=0;d<3;++d)if(!std::isfinite(freev[3*i+d]))return -2;}
 for(unsigned r=0;r<rows;++r){double sum=0,norm=0;
  for(unsigned d=0;d<3;++d){double n=normals[3*r+d];if(!std::isfinite(n))return -3;norm+=n*n;}
  if(std::abs(norm-1)>1e-12)return -3;
  for(unsigned j=0;j<4;++j){unsigned id=ids[4*r+j];double w=weights[4*r+j];if(id>=nodes||!std::isfinite(w))return -4;
   for(unsigned k=0;k<j;++k)if(id==ids[4*r+k])return -4;
   sum+=w;mobility[r]+=inverse[id]*w*w;}
  if(std::abs(sum)>1e-12||!(mobility[r]>0)||!std::isfinite(mobility[r]))return -4;
 }
 std::copy(freev,freev+3*nodes,velocity);std::fill(impulses,impulses+rows,0.);
 auto vn=[&](unsigned r){double value=0.;for(unsigned j=0;j<4;++j)for(unsigned d=0;d<3;++d)value+=weights[4*r+j]*normals[3*r+d]*velocity[3*ids[4*r+j]+d];return value;};
 auto add=[&](unsigned r,double amount){for(unsigned j=0;j<4;++j)for(unsigned d=0;d<3;++d){unsigned id=ids[4*r+j];velocity[3*id+d]+=inverse[id]*weights[4*r+j]*normals[3*r+d]*amount;}};
 for(unsigned step=1;step<=budget;++step){
  for(unsigned r=0;r<rows;++r){contact_normal::Output out{};
   // Unit algebraic scale: current_gap_m holds vn here; no position/time update.
   if(!contact_normal::project({vn(r),impulses[r],mobility[r],1.},out))return -5;
   impulses[r]=out.impulse;add(r,out.delta_impulse);
  }
  std::copy(freev,freev+3*nodes,velocity);for(unsigned r=0;r<rows;++r)add(r,impulses[r]);
  double scale=1e-30,residual=0.;for(unsigned r=0;r<rows;++r)scale=std::max(scale,impulses[r]);
  for(unsigned r=0;r<rows;++r){double value=vn(r);if(!std::isfinite(value))return -5;residual=std::max(residual,std::max(-value,std::abs(impulses[r]*value)/scale));}
  *iterations=step;if(residual<=tolerance)return 0;
 }
 return 1; // Budget exhaustion is an explicit nonaccepted candidate.
}
