// Keep classification in float while constructing and normalizing in double.
#define TOLERANCE_SWEEP_NO_MODULE
#include "tolerance_sweep.cpp"
#include "pseudonormal_trace.h"
#include <cstring>

struct MixedNormalMesh : PreparedMesh<float>
{
  using Base = PreparedMesh<float>;
  using DoubleMatrix = PreparedMesh<double>::Matrix;
  using Base::PreparedMesh;
  Eigen::Matrix<double,Eigen::Dynamic,3> source_face_normals;

  void use_double_normals(const nb::DRef<const DoubleMatrix> & normal_vertices)
  {
    if(normal_vertices.rows()!=vertices.rows() || normal_vertices.cols()!=3 ||
       !normal_vertices.allFinite() || normal_vertices.cast<float>()!=vertices)
      throw std::invalid_argument("Normal vertices must round to the float classifier geometry");
    const nb::DRef<const Eigen::MatrixXI> face_ref(faces);
    Eigen::Matrix<double,Eigen::Dynamic,3> fn,vn,en;
    Eigen::Matrix<Integer,Eigen::Dynamic,2> normal_edges;
    Eigen::VectorXI normal_edge_map;
    igl::per_face_normals(normal_vertices,face_ref,fn);
    igl::per_vertex_normals(normal_vertices,face_ref,
      igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,fn,vn);
    igl::per_edge_normals(normal_vertices,face_ref,
      igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,fn,en,normal_edges,normal_edge_map);
    if(normal_edges!=edges || normal_edge_map!=edge_map)
      throw std::runtime_error("Double normal construction changed the edge mapping");
    face_normals=fn.cast<float>();
    vertex_normals=vn.cast<float>();
    edge_normals=en.cast<float>();
    source_face_normals=fn;
  }

  auto normal_tables() const
  {
    return std::make_tuple(face_normals,vertex_normals,edge_normals,edges,edge_map,
      source_face_normals);
  }

  auto classify(
    const nb::DRef<const Matrix> & points,
    const nb::DRef<const Eigen::VectorXI> & indices,
    const nb::DRef<const Matrix> & closest,
    const float minimum_double_area,
    const float epsilon) const
  {
    return Base::classify(points,indices,closest,minimum_double_area,epsilon);
  }

  auto trace(
    const nb::DRef<const Matrix> & points,
    const nb::DRef<const Eigen::VectorXI> & indices,
    const nb::DRef<const Matrix> & closest,
    const float minimum_double_area,
    const float epsilon) const
  {
    auto expected=Base::classify(points,indices,closest,minimum_double_area,epsilon);
    Eigen::VectorXf signs(points.rows()),dots(points.rows());
    Eigen::VectorXd double_dots(points.rows());
    Matrix normals(points.rows(),3);
    Eigen::VectorXI kinds(points.rows()),features(points.rows()),small_areas(points.rows());
    const nb::DRef<const Matrix> vertex_ref(vertices);
    const nb::DRef<const Eigen::MatrixXI> face_ref(faces);
    for(int i=0;i<points.rows();++i)
    {
      Eigen::RowVector3f q=points.row(i),c=closest.row(i),n;
      float sign;
      int kind=-1,feature=-1;
      bool small_area=false;
      igl::pseudonormal_test_with_trace(vertex_ref,face_ref,face_normals,
        vertex_normals,edge_normals,edge_map,q,int(indices(i)),c,sign,n,
        minimum_double_area,epsilon,kind,feature,small_area);
      if(kind<0 || feature<0)
        throw std::runtime_error("Classifier did not select a traced normal");
      signs(i)=sign;
      normals.row(i)=n;
      kinds(i)=kind;
      features(i)=feature;
      small_areas(i)=small_area;
      dots(i)=(q-c).dot(n);
      double_dots(i)=(q.cast<double>()-c.cast<double>()).dot(n.cast<double>());
    }
    const auto & expected_normals=std::get<1>(expected);
    if(signs!=std::get<0>(expected) ||
       std::memcmp(normals.data(),expected_normals.data(),normals.size()*sizeof(float))!=0)
      throw std::runtime_error("Traced classifier differs from the uninstrumented classifier");
    return std::make_tuple(signs,normals,kinds,features,small_areas,dots,double_dots);
  }
};

NB_MODULE(_mixed_normals,m)
{
  nb::class_<MixedNormalMesh>(m,"PreparedMesh")
    .def(nb::init<const nb::DRef<const MixedNormalMesh::Matrix> &,
      const nb::DRef<const Eigen::MatrixXI> &>(),"vertices"_a.noconvert(),"faces"_a)
    .def("use_double_normals",&MixedNormalMesh::use_double_normals,"vertices"_a.noconvert())
    .def("normal_tables",&MixedNormalMesh::normal_tables)
    .def("classify",&MixedNormalMesh::classify,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"minimum_double_area"_a,"epsilon"_a)
    .def("trace",&MixedNormalMesh::trace,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"minimum_double_area"_a,"epsilon"_a);
}
