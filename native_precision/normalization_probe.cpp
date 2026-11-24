// Diagnose final pseudonormal normalization without changing the upstream API.
#define TOLERANCE_SWEEP_NO_MODULE
#include "tolerance_sweep.cpp"
#include "unnormalized_normals.h"

template <typename Scalar>
struct NormalizationProbe : PreparedMesh<Scalar>
{
  using Base = PreparedMesh<Scalar>;
  using Matrix = typename Base::Matrix;
  using NormalMatrix = Eigen::Matrix<Scalar,Eigen::Dynamic,3>;
  NormalMatrix normalized_vertex,normalized_edge,raw_vertex,raw_edge;

  NormalizationProbe(
    const nb::DRef<const Matrix> & vertices,
    const nb::DRef<const Eigen::MatrixXI> & faces): Base(vertices,faces)
  {
    normalized_vertex=this->vertex_normals;
    normalized_edge=this->edge_normals;
    igl::per_vertex_normals_unnormalized(vertices,faces,
      igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,this->face_normals,raw_vertex);
    Eigen::Matrix<Integer,Eigen::Dynamic,2> edges;
    Eigen::VectorXI edge_map;
    igl::per_edge_normals_unnormalized(vertices,faces,
      igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,this->face_normals,
      raw_edge,edges,edge_map);
    if(edges!=this->edges || edge_map!=this->edge_map)
      throw std::runtime_error("Normal probe changed the edge mapping");
  }

  auto normals() const
  {
    return std::make_tuple(this->face_normals,normalized_vertex,normalized_edge,
      raw_vertex,raw_edge,this->edges,this->edge_map);
  }

  auto normalization_checks() const
  {
    return std::make_tuple(
      raw_vertex.rowwise().normalized().eval(),
      raw_edge.rowwise().normalized().eval(),
      raw_vertex.rowwise().squaredNorm().eval(),
      raw_edge.rowwise().squaredNorm().eval());
  }

  auto classify(
    const nb::DRef<const Matrix> & points,
    const nb::DRef<const Eigen::VectorXI> & indices,
    const nb::DRef<const Matrix> & closest,
    const Scalar minimum_double_area,
    const Scalar epsilon,
    const bool normalize_vertex,
    const bool normalize_edge)
  {
    this->vertex_normals=normalize_vertex ? normalized_vertex : raw_vertex;
    this->edge_normals=normalize_edge ? normalized_edge : raw_edge;
    return Base::classify(points,indices,closest,minimum_double_area,epsilon);
  }
};

template <typename Scalar>
void bind_probe(nb::module_ & module,const char * name)
{
  using Probe = NormalizationProbe<Scalar>;
  nb::class_<Probe>(module,name)
    .def(nb::init<const nb::DRef<const typename Probe::Matrix> &,
      const nb::DRef<const Eigen::MatrixXI> &>(),"vertices"_a.noconvert(),"faces"_a)
    .def("normals",&Probe::normals)
    .def("normalization_checks",&Probe::normalization_checks)
    .def("classify",&Probe::classify,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"minimum_double_area"_a,"epsilon"_a,
      "normalize_vertex"_a=true,"normalize_edge"_a=true);
}

NB_MODULE(_normalization_probe,m)
{
  bind_probe<float>(m,"PreparedMesh");
  bind_probe<double>(m,"PreparedMeshDouble");
}
