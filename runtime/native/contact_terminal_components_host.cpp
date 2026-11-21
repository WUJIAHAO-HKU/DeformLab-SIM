// Host-only component-local normal projection using the shared primitive.
#include "contact_normal_history.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
struct TerminalComponent{std::vector<unsigned> rows,nodes;bool complete=false;};
extern "C" int terminal_components(unsigned nodes,unsigned rows,const unsigned* ids,const double* weights,
 const double* normals,const double* inverse,const double* freev,unsigned budget,double tolerance,
 double* velocity,double* impulses,unsigned* iterations,unsigned long long* counts){
 if(!nodes||!rows||!ids||!weights||!normals||!inverse||!freev||!velocity||!impulses||!iterations||!counts||!budget||!(tolerance>0)||!std::isfinite(tolerance))return -1;
 std::vector<double> mobility(rows);std::vector<unsigned> parent(rows),owner(nodes,rows);std::iota(parent.begin(),parent.end(),0);
 auto root=[&](unsigned x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
 for(unsigned i=0;i<nodes;++i){if(!std::isfinite(inverse[i])||inverse[i]<0)return -2;for(unsigned d=0;d<3;++d)if(!std::isfinite(freev[3*i+d]))return -2;}
 for(unsigned r=0;r<rows;++r){double sum=0,norm=0;
  for(unsigned d=0;d<3;++d){double n=normals[3*r+d];if(!std::isfinite(n))return -3;norm+=n*n;}
  if(std::abs(norm-1)>1e-12)return -3;
  for(unsigned j=0;j<4;++j){unsigned id=ids[4*r+j];double w=weights[4*r+j];if(id>=nodes||!std::isfinite(w))return -4;
   for(unsigned k=0;k<j;++k)if(id==ids[4*r+k])return -4;
   sum+=w;mobility[r]+=inverse[id]*w*w;
   if(w!=0&&inverse[id]>0){if(owner[id]==rows)owner[id]=r;else parent[root(r)]=root(owner[id]);}
  }
  if(std::abs(sum)>1e-12||!(mobility[r]>0)||!std::isfinite(mobility[r]))return -4;
 }
 std::vector<TerminalComponent> components;std::vector<unsigned> mapping(rows,rows);
 for(unsigned r=0;r<rows;++r){unsigned representative=root(r);if(mapping[representative]==rows){mapping[representative]=components.size();components.emplace_back();}components[mapping[representative]].rows.push_back(r);}
 for(unsigned i=0;i<nodes;++i)if(owner[i]!=rows)components[mapping[root(owner[i])]].nodes.push_back(i);
 std::copy(freev,freev+3*nodes,velocity);std::fill(impulses,impulses+rows,0.);std::fill(counts,counts+4,0ull);counts[0]=components.size();
 for(const auto& c:components)counts[1]=std::max(counts[1],static_cast<unsigned long long>(c.rows.size()));
 auto vn=[&](unsigned r){double value=0.;for(unsigned j=0;j<4;++j)for(unsigned d=0;d<3;++d)value+=weights[4*r+j]*normals[3*r+d]*velocity[3*ids[4*r+j]+d];return value;};
 auto add=[&](unsigned r,double amount){for(unsigned j=0;j<4;++j)for(unsigned d=0;d<3;++d){unsigned id=ids[4*r+j];velocity[3*id+d]+=inverse[id]*weights[4*r+j]*normals[3*r+d]*amount;}};
 for(unsigned step=1;step<=budget;++step){bool all=true;
  for(auto& c:components){if(c.complete)continue;counts[2]+=c.rows.size();++counts[3];
   for(unsigned r:c.rows){contact_normal::Output out{};if(!contact_normal::project({vn(r),impulses[r],mobility[r],1.},out))return -5;impulses[r]=out.impulse;add(r,out.delta_impulse);}
   // Components share no writable velocity DOFs. Fixed supports are read-only.
   for(unsigned i:c.nodes)for(unsigned d=0;d<3;++d)velocity[3*i+d]=freev[3*i+d];
   for(unsigned r:c.rows)add(r,impulses[r]);
   double scale=1e-30,residual=0.;for(unsigned r:c.rows)scale=std::max(scale,impulses[r]);
   for(unsigned r:c.rows){double value=vn(r);if(!std::isfinite(value))return -5;residual=std::max(residual,std::max(-value,std::abs(impulses[r]*value)/scale));}
   c.complete=residual<=tolerance;all &=c.complete;
  }
  *iterations=step;if(all)return 0;
 }
 return 1;
}
