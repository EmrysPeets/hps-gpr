// Single-threaded magnetic transport. Units: mm, MeV/c, tesla; charge in units of e.
#include <cmath>
#include <cstdint>
#include <algorithm>
struct V { double x,y,z; V operator+(V a)const{return{x+a.x,y+a.y,z+a.z};} V operator-(V a)const{return{x-a.x,y-a.y,z-a.z};} V operator*(double f)const{return{x*f,y*f,z*f};} };
static inline double dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static inline V cross(V a,V b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
static inline V vec(const double*p){return{p[0],p[1],p[2]};}
struct Field {
 const double*v; int nx,ny,nz; V lo,ds;
 V at(V r)const{
  double x=(r.x-lo.x)/ds.x,y=(r.y-lo.y)/ds.y,z=(r.z-lo.z)/ds.z;
  if(x<0||y<0||z<0||x>nx-1||y>ny-1||z>nz-1)return{0,0,0};
  int i=std::min((int)x,nx-2),j=std::min((int)y,ny-2),k=std::min((int)z,nz-2);
  x-=i;y-=j;z-=k;V b{0,0,0};
  for(int a=0;a<2;a++)for(int c=0;c<2;c++)for(int d=0;d<2;d++){
   double w=(a?x:1-x)*(c?y:1-y)*(d?z:1-z);
   const double*p=v+3*(((i+a)*ny+j+c)*nz+k+d);b=b+vec(p)*w;
  }return b;
 }
};
struct State{V r,u;};
static inline State deriv(State a,const Field&f,double k){return{a.u,cross(a.u,f.at(a.r))*k};}
static inline State plus(State a,State b,double t){return{a.r+b.r*t,a.u+b.u*t};}
static State step(State a,const Field&f,double k,double h){
 State b=deriv(a,f,k),c=deriv(plus(a,b,h/2),f,k),d=deriv(plus(a,c,h/2),f,k),e=deriv(plus(a,d,h),f,k);
 State v{a.r+(b.r+c.r*2+d.r*2+e.r)*(h/6),a.u+(b.u+c.u*2+d.u*2+e.u)*(h/6)};
 v.u=v.u*(1/std::sqrt(dot(v.u,v.u)));return v;
}
static V hermite(State a,State b,double h,double t){
 double t2=t*t,t3=t2*t;
 return a.r*(2*t3-3*t2+1)+a.u*(h*(t3-2*t2+t))+b.r*(-2*t3+3*t2)+b.u*(h*(t3-t2));
}
static V hermite_dt(State a,State b,double h,double t){
 return a.r*(6*t*t-6*t)+a.u*(h*(3*t*t-4*t+1))+b.r*(-6*t*t+6*t)+b.u*(h*(3*t*t-2*t));
}
// Sensor row: center(3), normal(3), a(3), b(3), halfwidths(2), zmin,zmax, station0,half0top,view0axial.
static void hit_segment(State a,State b,double h,const double*s,int ns,uint16_t masks[4]){
 for(int j=0;j<ns;j++){
  const double*p=s+21*j;
  if(b.r.z<p[14]-0.02||a.r.z>p[15]+0.02)continue;
  V center=vec(p),n=vec(p+3);double da=dot(a.r-center,n),db=dot(b.r-center,n);
  if(da*db>0||std::abs(da-db)<1e-15)continue;
  double t=da/(da-db);
  for(int k=0;k<4;k++){
   double de=dot(hermite_dt(a,b,h,t),n);
   if(std::abs(de)<1e-15)break;
   t=std::clamp(t-dot(hermite(a,b,h,t)-center,n)/de,0.,1.);
  }
  if(std::abs(dot(hermite(a,b,h,t)-center,n))>1e-7){
   double lo=0.,hi=1.;
   for(int k=0;k<40;k++){
    t=(lo+hi)/2;double dm=dot(hermite(a,b,h,t)-center,n);
    if(da*dm<=0)hi=t;else lo=t;
   }
  }
  V d=hermite(a,b,h,t)-center;
  if(std::abs(dot(d,n))>1e-7)continue;
  if(std::abs(dot(d,vec(p+6)))<=p[12]+1e-9&&std::abs(dot(d,vec(p+9)))<=p[13]+1e-9)
   masks[2*(int)p[17]+(int)p[18]]|=uint16_t(1u<<(int)p[16]);
 }
}
static int run_one(V momentum,V origin,int charge,const Field&f,const double*s,int ns,double maxstep,double maxB,double zstop,uint16_t*out,double*path,int capacity){
 double pmag=std::sqrt(dot(momentum,momentum));State a{origin,momentum*(1/pmag)};
 uint16_t masks[4]={0,0,0,0};double distance=0.;int iteration=0,written=0;
 auto save=[&](){if(path&&written<capacity){double*p=path+7*written;p[0]=distance;p[1]=a.r.x;p[2]=a.r.y;p[3]=a.r.z;p[4]=a.u.x;p[5]=a.u.y;p[6]=a.u.z;++written;}};
 save();
 double h=std::min(maxstep,0.01*pmag/(0.299792458*std::max(maxB,1e-12)));
 while(a.r.z<zstop&&a.u.z>0&&distance<6000&&iteration<50000){
  State b=step(a,f,0.299792458*charge/pmag,h);
  hit_segment(a,b,h,s,ns,masks);a=b;distance+=h;++iteration;save();
  // Outside the finite field grid with outward motion, B=0 forever; these
  // transverse grid boundaries enclose every active sensor in this study.
  if((a.r.x<f.lo.x&&a.u.x<=0)||(a.r.x>f.lo.x+(f.nx-1)*f.ds.x&&a.u.x>=0)||
     (a.r.y<f.lo.y&&a.u.y<=0)||(a.r.y>f.lo.y+(f.ny-1)*f.ds.y&&a.u.y>=0))break;
 }
 out[0]=(masks[0]|masks[2]);out[1]=(masks[1]|masks[3]);
 out[2]=(masks[0]&masks[1])|(masks[2]&masks[3]);
 out[3]=iteration>=50000||distance>=6000?1:0;
 return written;
}
extern "C" void transport(const double*mom,int count,const double*origin,int charge,const double*values,const int*dims,const double*grid,const double*sensors,int ns,double maxstep,double maxB,double zstop,uint16_t*out){
 Field f{values,dims[0],dims[1],dims[2],vec(grid),vec(grid+3)};
 for(int i=0;i<count;i++)run_one(vec(mom+3*i),vec(origin),charge,f,sensors,ns,maxstep,maxB,zstop,out+4*i,nullptr,0);
}
extern "C" int trajectory(const double*mom,const double*origin,int charge,const double*values,const int*dims,const double*grid,const double*sensors,int ns,double maxstep,double maxB,double zstop,uint16_t*out,double*path,int capacity){
 Field f{values,dims[0],dims[1],dims[2],vec(grid),vec(grid+3)};
 return run_one(vec(mom),vec(origin),charge,f,sensors,ns,maxstep,maxB,zstop,out,path,capacity);
}
