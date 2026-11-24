// Trace the double classifier without changing its arithmetic or normal tables.
#define TOLERANCE_SWEEP_NO_MODULE
#include "tolerance_sweep.cpp"
#include "pseudonormal_trace.h"
#include <igl/point_simplex_squared_distance.h>
#include <cstring>

struct DoubleDiagnostics : PreparedMesh<double>
{
  using Base=PreparedMesh<double>;
  using Base::PreparedMesh;

  auto normal_tables() const
  {
    return std::make_tuple(face_normals,vertex_normals,edge_normals,edges,edge_map);
  }

  auto trace(
    const nb::DRef<const Matrix> & points,
    const nb::DRef<const Eigen::VectorXI> & indices,
    const nb::DRef<const Matrix> & closest,
    const double area_threshold,
    const double epsilon) const
  {
    const auto expected=Base::classify(points,indices,closest,area_threshold,epsilon);
    const nb::DRef<const Matrix> vertex_ref(vertices);
    const nb::DRef<const Eigen::MatrixXI> face_ref(faces);
    Eigen::VectorXd signs(points.rows()),areas(points.rows()),dots(points.rows());
    Matrix normals(points.rows(),3),barycentrics(points.rows(),3);
    Matrix nearest_barycentrics(points.rows(),3),nearest_points(points.rows(),3);
    Eigen::VectorXI kinds(points.rows()),features(points.rows()),small(points.rows());
    for(int i=0;i<points.rows();++i)
    {
      Eigen::RowVector3d q=points.row(i),c=closest.row(i),n,bary,nearest,nearest_bary;
      int kind=-1,feature=-1;
      bool small_area=false;
      double sign,squared_distance;
      igl::pseudonormal_test_with_trace(vertex_ref,face_ref,face_normals,
        vertex_normals,edge_normals,edge_map,q,int(indices(i)),c,sign,n,
        area_threshold,epsilon,kind,feature,small_area);
      if(kind<0 || feature<0)
        throw std::runtime_error("Classifier did not select a traced normal");
      const Eigen::RowVector3d a=vertices.row(faces(indices(i),0));
      const Eigen::RowVector3d b=vertices.row(faces(indices(i),1));
      const Eigen::RowVector3d d=vertices.row(faces(indices(i),2));
      Eigen::Matrix<double,1,1> area;
      igl::doublearea(a,b,d,area);
      igl::barycentric_coordinates(c,a,b,d,bary);
      igl::point_simplex_squared_distance<3>(q,vertex_ref,face_ref,int(indices(i)),
        squared_distance,nearest,nearest_bary);
      signs(i)=sign; normals.row(i)=n; dots(i)=(q-c).dot(n);
      areas(i)=area(0); kinds(i)=kind; features(i)=feature; small(i)=small_area;
      barycentrics.row(i)=bary; nearest_barycentrics.row(i)=nearest_bary;
      nearest_points.row(i)=nearest;
    }
    const auto & expected_normals=std::get<1>(expected);
    if(signs!=std::get<0>(expected) ||
       std::memcmp(normals.data(),expected_normals.data(),normals.size()*sizeof(double))!=0)
      throw std::runtime_error("Trace changed the original classifier outputs");
    return std::make_tuple(signs,normals,kinds,features,small,areas,barycentrics,
      dots,nearest_barycentrics,nearest_points);
  }
};

NB_MODULE(_double_diagnostics,m)
{
  nb::class_<DoubleDiagnostics>(m,"PreparedMesh")
    .def(nb::init<const nb::DRef<const DoubleDiagnostics::Matrix> &,
      const nb::DRef<const Eigen::MatrixXI> &>(),"vertices"_a.noconvert(),"faces"_a)
    .def("normal_tables",&DoubleDiagnostics::normal_tables)
    .def("trace",&DoubleDiagnostics::trace,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"area_threshold"_a,"epsilon"_a);
}
