// Host-only component-local Signorini/Coulomb iteration using shared primitives.
// freev excludes the numerical warm guess; the guess is applied exactly once.
#include "contact_normal_history.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
struct TerminalComponent{std::vector<unsigned> rows,nodes;bool complete=false;};
extern "C" int terminal_coulomb(unsigned nodes,unsigned rows,const unsigned* ids,const double* weights,
 const double* normals,const double* inverse,const double* freev,const double* friction,
 const double* warm_normal,const double* warm_tangent,unsigned budget,double tolerance,
 double* velocity,double* impulses,double* tangents,unsigned* iterations,unsigned long long* counts){
 if(!nodes||!rows||!ids||!weights||!normals||!inverse||!freev||!friction||!velocity||!impulses||!tangents||((warm_normal==nullptr)!=(warm_tangent==nullptr))||!iterations||!counts||!budget||!(tolerance>0)||!std::isfinite(tolerance))return -1;
 std::vector<double> mobility(rows);std::vector<unsigned> parent(rows),owner(nodes,rows);std::iota(parent.begin(),parent.end(),0);
 auto root=[&](unsigned x){while(parent[x]!=x){parent[x]=parent[parent[x]];x=parent[x];}return x;};
 for(unsigned i=0;i<nodes;++i){if(!std::isfinite(inverse[i])||inverse[i]<0)return -2;for(unsigned d=0;d<3;++d)if(!std::isfinite(freev[3*i+d]))return -2;}
 for(unsigned r=0;r<rows;++r){double sum=0,norm=0;
  for(unsigned d=0;d<3;++d){double n=normals[3*r+d];if(!std::isfinite(n))return -3;norm+=n*n;}
  if(std::abs(norm-1)>1e-12||!std::isfinite(friction[r])||friction[r]<0)return -3;
  for(unsigned j=0;j<4;++j){unsigned id=ids[4*r+j];double w=weights[4*r+j];if(id>=nodes||!std::isfinite(w))return -4;
   for(unsigned k=0;k<j;++k)if(id==ids[4*r+k])return -4;
   sum+=w;mobility[r]+=inverse[id]*w*w;
   if(w!=0&&inverse[id]>0){if(owner[id]==rows)owner[id]=r;else parent[root(r)]=root(owner[id]);}
  }
  if(std::abs(sum)>1e-12||!(mobility[r]>0)||!std::isfinite(mobility[r]))return -4;
  if(warm_normal){contact_tangent::Output trial{};contact_tangent::Input in{{normals[3*r],normals[3*r+1],normals[3*r+2]},{0,0,0},{warm_tangent[3*r],warm_tangent[3*r+1],warm_tangent[3*r+2]},mobility[r],friction[r],warm_normal[r]};if(!contact_tangent::project(in,trial))return -4;}
 }
 std::vector<TerminalComponent> components;std::vector<unsigned> mapping(rows,rows);
 for(unsigned r=0;r<rows;++r){unsigned representative=root(r);if(mapping[representative]==rows){mapping[representative]=components.size();components.emplace_back();}components[mapping[representative]].rows.push_back(r);}
 for(unsigned i=0;i<nodes;++i)if(owner[i]!=rows)components[mapping[root(owner[i])]].nodes.push_back(i);
 std::copy(freev,freev+3*nodes,velocity);
 for(unsigned r=0;r<rows;++r){impulses[r]=warm_normal?warm_normal[r]:0.;for(unsigned d=0;d<3;++d)tangents[3*r+d]=warm_tangent?warm_tangent[3*r+d]:0.;}std::fill(counts,counts+4,0ull);counts[0]=components.size();
 for(const auto& c:components)counts[1]=std::max(counts[1],static_cast<unsigned long long>(c.rows.size()));
 using V=contact_tangent::Vec;
 auto normal=[&](unsigned r){return V{normals[3*r],normals[3*r+1],normals[3*r+2]};};
 auto tangent=[&](unsigned r){return V{tangents[3*r],tangents[3*r+1],tangents[3*r+2]};};
 auto relative=[&](unsigned r){V value{0,0,0};for(unsigned j=0;j<4;++j){unsigned id=ids[4*r+j];value=contact_tangent::add(value,contact_tangent::mul({velocity[3*id],velocity[3*id+1],velocity[3*id+2]},weights[4*r+j]));}return value;};
 auto add=[&](unsigned r,V amount){for(unsigned j=0;j<4;++j){unsigned id=ids[4*r+j];double a=inverse[id]*weights[4*r+j];velocity[3*id]+=a*amount.x;velocity[3*id+1]+=a*amount.y;velocity[3*id+2]+=a*amount.z;}};
 for(unsigned r=0;r<rows;++r)add(r,contact_tangent::add(contact_tangent::mul(normal(r),impulses[r]),tangent(r)));
 double global_speed=1e-30;for(unsigned i=0;i<nodes;++i)global_speed=std::max(global_speed,contact_tangent::norm({freev[3*i],freev[3*i+1],freev[3*i+2]}));
 for(unsigned step=1;step<=budget;++step){bool all=true;
  for(auto& c:components){if(c.complete)continue;counts[2]+=c.rows.size();++counts[3];
   for(unsigned r:c.rows){contact_normal::Output out{};double vn=contact_tangent::dot(relative(r),normal(r));
    if(!contact_normal::project({vn,impulses[r],mobility[r],1.},out))return -5;
    impulses[r]=out.impulse;add(r,contact_tangent::mul(normal(r),out.delta_impulse));
    contact_tangent::Output next{};
    if(!contact_tangent::project({normal(r),relative(r),tangent(r),mobility[r],friction[r],impulses[r]},next))return -5;
    add(r,next.delta_impulse);tangents[3*r]=next.impulse.x;tangents[3*r+1]=next.impulse.y;tangents[3*r+2]=next.impulse.z;
   }
   // Components share no writable velocity DOFs. Fixed supports are read-only.
   for(unsigned i:c.nodes)for(unsigned d=0;d<3;++d)velocity[3*i+d]=freev[3*i+d];
   for(unsigned r:c.rows)add(r,contact_tangent::add(contact_tangent::mul(normal(r),impulses[r]),tangent(r)));
   double scale=1e-30,normal_scale=1e-30,residual=0.;
   for(unsigned r:c.rows){normal_scale=std::max(normal_scale,impulses[r]);scale=std::max(scale,std::max(impulses[r],friction[r]*impulses[r]));}
   for(unsigned r:c.rows){V rv=relative(r);double vn=contact_tangent::dot(rv,normal(r));if(!std::isfinite(vn))return -5;
    contact_tangent::Output check{};if(!contact_tangent::project({normal(r),rv,tangent(r),mobility[r],friction[r],impulses[r]},check))return -5;
    residual=std::max(residual,std::max(-vn,std::abs(impulses[r]*vn)/normal_scale));
    residual=std::max(residual,contact_tangent::norm(check.delta_impulse)/scale*global_speed);
   }
   c.complete=residual<=tolerance;all &=c.complete;
  }
  *iterations=step;if(all)return 0;
 }
 return 1;
}
