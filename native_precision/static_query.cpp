#ifndef IGL_STATIC_LIBRARY
#error "Compile with IGL_STATIC_LIBRARY to exercise the precompiled signatures"
#endif

#include <igl/signed_distance.h>
#include <cmath>
#include <iostream>
#include <limits>
template<typename Scalar,int Dim,int Columns,int Options> bool check_bounded()
{
  using Matrix = Eigen::Matrix<Scalar,Eigen::Dynamic,Columns,Options>;
  using Faces = Eigen::Matrix<int,Eigen::Dynamic,Columns,Options>;
  using Row = Eigen::Matrix<Scalar,1,Dim>;
  Matrix v=Matrix::Zero(Dim,Dim);
  Faces f(1,Dim);
  for(int j=0;j<Dim;++j)
  {
    f(j)=j;
    if(j>0) v(j,j-1)=1;
  }
  igl::AABB<Matrix,Dim> tree;
  tree.init(v,f);
  Row p=Row::Constant(Scalar(.25));
  p(Dim-1)=1;
  Row c;
  int nearest=-1;
  const Scalar squared_distance=tree.squared_distance(v,f,p,Scalar(0),Scalar(2),nearest,c);
  Row expected=p;
  expected(Dim-1)=0;
  return squared_distance==Scalar(1) && nearest==0 && c==expected;
}
template<typename Scalar> bool check()
{
  Eigen::Matrix<Scalar,Eigen::Dynamic,3> v(4,3),p(4,3),c,n;
  Eigen::Matrix<int,Eigen::Dynamic,3> f(4,3);
  f << 0,1,3, 0,2,1, 0,3,2, 1,2,3;
  for(Scalar scale: {Scalar(1),Scalar(1)/Scalar(1024)})
  {
    v << 0,0,0, 1,0,0, 0,1,0, 0,0,1;
    p << Scalar(.125),Scalar(.125),Scalar(.125), Scalar(.25),Scalar(.25),-1, Scalar(.5),-1,-1, -1,-1,-1;
    v*=scale; p*=scale;
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> s;
    Eigen::VectorXi i;
    igl::signed_distance(p,v,f,igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
      -std::numeric_limits<Scalar>::infinity(),std::numeric_limits<Scalar>::infinity(),s,i,c,n);
    Eigen::Matrix<Scalar,4,1> expected;
    expected << -Scalar(.125),1,std::sqrt(Scalar(2)),std::sqrt(Scalar(3));
    if(!c.allFinite() || !n.allFinite() || (s-expected*scale).norm()>Scalar(16)*std::numeric_limits<Scalar>::epsilon()*scale)
      return false;
  }
  return true;
}
int main()
{
  if(!check<float>() || !check<double>()) return 1;
  if(!check_bounded<double,2,Eigen::Dynamic,Eigen::ColMajor>() ||
     !check_bounded<double,3,Eigen::Dynamic,Eigen::ColMajor>() ||
     !check_bounded<float,3,3,Eigen::RowMajor>()) return 1;
  std::cout<<"Static float and double signed-distance queries passed\n";
}
