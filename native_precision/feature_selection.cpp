// Replay cached nearest triangles through the production barycentric classifier.
#define TOLERANCE_SWEEP_NO_MODULE
#include "tolerance_sweep.cpp"
#include <igl/point_simplex_squared_distance.h>
#include <igl/pseudonormal_test.h>

template <typename Scalar>
struct FeatureSelection : PreparedMesh<Scalar>
{
  using Base=PreparedMesh<Scalar>;
  using Matrix=typename Base::Matrix;
  using Base::PreparedMesh;

  auto replay(
    const nb::DRef<const Matrix> & points,
    const nb::DRef<const Eigen::VectorXI> & indices) const
  {
    if(points.cols()!=3 || points.rows()==0 || indices.size()!=points.rows() ||
       !points.allFinite() || indices.minCoeff()<0 || indices.maxCoeff()>=this->faces.rows())
      throw std::invalid_argument("Expected finite queries and valid cached triangle indices");
    const nb::DRef<const Matrix> vertex_ref(this->vertices);
    const nb::DRef<const Eigen::MatrixXI> face_ref(this->faces);
    Eigen::Matrix<Scalar,Eigen::Dynamic,1> signs(points.rows()),squared_distance(points.rows());
    Matrix closest(points.rows(),3),normals(points.rows(),3),barycentric(points.rows(),3);
    igl::parallel_for(points.rows(),[&](const int i)
    {
      Eigen::Matrix<Scalar,1,3> q=points.row(i),c,n,b;
      Scalar d,s;
      igl::point_simplex_squared_distance<3>(q,vertex_ref,face_ref,int(indices(i)),d,c,b);
      igl::pseudonormal_test(vertex_ref,face_ref,this->face_normals,
        this->vertex_normals,this->edge_normals,this->edge_map,q,int(indices(i)),c,b,s,n);
      signs(i)=s; squared_distance(i)=d;
      closest.row(i)=c; normals.row(i)=n; barycentric.row(i)=b;
    },10000);
    return std::make_tuple(signs,normals,closest,barycentric,squared_distance);
  }
};

NB_MODULE(_feature_selection,m)
{
  nb::class_<FeatureSelection<float>>(m,"PreparedMeshFloat")
    .def(nb::init<const nb::DRef<const FeatureSelection<float>::Matrix> &,
      const nb::DRef<const Eigen::MatrixXI> &>(),"vertices"_a.noconvert(),"faces"_a)
    .def("classify",&FeatureSelection<float>::classify,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"minimum_double_area"_a,"epsilon"_a)
    .def("replay",&FeatureSelection<float>::replay,"points"_a.noconvert(),"indices"_a);
  nb::class_<FeatureSelection<double>>(m,"PreparedMeshDouble")
    .def(nb::init<const nb::DRef<const FeatureSelection<double>::Matrix> &,
      const nb::DRef<const Eigen::MatrixXI> &>(),"vertices"_a.noconvert(),"faces"_a)
    .def("classify",&FeatureSelection<double>::classify,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"minimum_double_area"_a,"epsilon"_a)
    .def("replay",&FeatureSelection<double>::replay,"points"_a.noconvert(),"indices"_a);
}
