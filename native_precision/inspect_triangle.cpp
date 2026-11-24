// Inspect libigl's area and barycentric calculations for one cached triangle.
#include <igl/barycentric_coordinates.h>
#include <igl/doublearea.h>
#include <iomanip>
#include <iostream>

template <typename Scalar>
void inspect(const Eigen::Matrix<double,3,3> & triangle,
  const Eigen::RowVector3d & closest)
{
  const Eigen::Matrix<Scalar,1,3> a=triangle.row(0).cast<Scalar>();
  const Eigen::Matrix<Scalar,1,3> b=triangle.row(1).cast<Scalar>();
  const Eigen::Matrix<Scalar,1,3> c=triangle.row(2).cast<Scalar>();
  const Eigen::Matrix<Scalar,1,3> p=closest.cast<Scalar>();
  Eigen::Matrix<Scalar,1,3> barycentric;
  Eigen::Matrix<Scalar,1,1> area;
  igl::doublearea(a,b,c,area);
  igl::barycentric_coordinates(p,a,b,c,barycentric);
  std::cout << "{\"twice_area\":" << area(0) << ",\"barycentric\":[";
  for(int i=0;i<3;++i)
  {
    if(i) std::cout << ',';
    if(std::isfinite(barycentric(i))) std::cout << barycentric(i);
    else std::cout << "null";
  }
  std::cout << "]}";
}

int main()
{
  Eigen::Matrix<double,3,3> triangle;
  Eigen::RowVector3d closest_float,closest_double;
  for(int i=0;i<3;++i)
    for(int j=0;j<3;++j)
      if(!(std::cin >> triangle(i,j))) return 1;
  for(int i=0;i<3;++i)
    if(!(std::cin >> closest_float(i))) return 1;
  for(int i=0;i<3;++i)
    if(!(std::cin >> closest_double(i))) return 1;
  std::cout << std::setprecision(17) << "{\"float\":";
  inspect<float>(triangle,closest_float);
  std::cout << ",\"double_on_float_inputs\":";
  inspect<double>(triangle.cast<float>().cast<double>(),closest_float);
  std::cout << ",\"double_on_original_inputs\":";
  inspect<double>(triangle,closest_double);
  std::cout << "}\n";
}
