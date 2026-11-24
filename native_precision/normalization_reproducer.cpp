// Reproduce the cached nonzero float normal's vectorized normalization failure.
#include <Eigen/Core>
#include <cstdio>

int main()
{
  Eigen::Matrix<float,Eigen::Dynamic,3> normals(8,3);
  for(int i=0;i<normals.rows();++i)
    normals.row(i) << 2.6072526112518294e-20f,1.5881867761018131e-21f,0.0f;
  normals.row(0).setZero();
  const Eigen::VectorXf squared_lengths=normals.rowwise().squaredNorm();
  const Eigen::VectorXf lengths=normals.rowwise().norm();
  const float scalar_length=normals.row(4).norm();
  normals.rowwise().normalize();
  std::printf("EIGEN_FAST_MATH=%d\nsquared_length=%.9g\nrowwise_length=%.9g\nscalar_length=%.9g\n",
    EIGEN_FAST_MATH,squared_lengths(4),lengths(4),scalar_length);
  std::printf("normalized_nonzero=[%.9g, %.9g, %.9g]\nnormalized_zero=[%.9g, %.9g, %.9g]\n",
    normals(4,0),normals(4,1),normals(4,2),normals(0,0),normals(0,1),normals(0,2));
  // These checks discriminate the observed SIMD subnormal behavior from the
  // separate unconditional division by zero for an exactly zero vector.
  if(!(squared_lengths(4)>0 && scalar_length>0) || normals.row(0).allFinite())
    return 1;
#if EIGEN_FAST_MATH
  return (lengths(4)==0 && !normals.row(4).allFinite()) ? 0 : 2;
#else
  return (lengths(4)>0 && normals.row(4).allFinite()) ? 0 : 3;
#endif
}
