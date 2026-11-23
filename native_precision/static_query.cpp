#ifndef IGL_STATIC_LIBRARY
#error "Compile with IGL_STATIC_LIBRARY to exercise the precompiled signature"
#endif

#include <igl/signed_distance.h>
#include <cmath>
#include <iostream>
#include <limits>
int main()
{
  Eigen::Matrix<float,Eigen::Dynamic,3> v(4,3),p(4,3),c,n;
  Eigen::Matrix<int,Eigen::Dynamic,3> f(4,3);
  f << 0,1,3, 0,2,1, 0,3,2, 1,2,3;
  for(float scale: {1.0f,1.0f/1024.0f})
  {
    v << 0,0,0, 1,0,0, 0,1,0, 0,0,1;
    p << .125f,.125f,.125f, .25f,.25f,-1, .5f,-1,-1, -1,-1,-1;
    v*=scale; p*=scale;
    Eigen::VectorXf s;
    Eigen::VectorXi i;
    igl::signed_distance(p,v,f,igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
      -std::numeric_limits<float>::infinity(),std::numeric_limits<float>::infinity(),s,i,c,n);
    Eigen::Vector4f expected;
    expected << -.125f,1,std::sqrt(2.0f),std::sqrt(3.0f);
    if(!c.allFinite() || !n.allFinite() || (s-expected*scale).norm()>1e-6f*scale)
    { std::cerr<<"static float query failed\n"; return 1; }
  }
  std::cout<<"Static float pseudonormal queries passed\n";
}
