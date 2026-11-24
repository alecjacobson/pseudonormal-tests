// Temporary experiment interface; no changes to libigl's public API.
#include "default_types.h"
#include "pseudonormal_runtime.h"
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/parallel_for.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/tuple.h>
#include <cmath>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;
using FloatMatrix = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Options>;

struct PreparedMesh
{
  FloatMatrix vertices;
  Eigen::MatrixXI faces;
  Eigen::Matrix<float,Eigen::Dynamic,3> face_normals,vertex_normals,edge_normals;
  Eigen::Matrix<Integer,Eigen::Dynamic,2> edges;
  Eigen::VectorXI edge_map;

  PreparedMesh(
    const nb::DRef<const FloatMatrix> & v,
    const nb::DRef<const Eigen::MatrixXI> & f): vertices(v),faces(f)
  {
    if(v.cols()!=3 || f.cols()!=3 || v.rows()==0 || f.rows()==0 ||
       !v.allFinite() || f.minCoeff()<0 || f.maxCoeff()>=v.rows())
      throw std::invalid_argument("Expected finite 3D vertices and valid triangles");
    // Match the signed_distance_3 normal construction and its matrix types.
    igl::per_face_normals(v,f,face_normals);
    igl::per_vertex_normals(v,f,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,
      face_normals,vertex_normals);
    igl::per_edge_normals(v,f,igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,
      face_normals,edge_normals,edges,edge_map);
  }

  auto classify(
    const nb::DRef<const FloatMatrix> & points,
    const nb::DRef<const Eigen::VectorXI> & indices,
    const nb::DRef<const FloatMatrix> & closest,
    const float minimum_double_area,
    const float epsilon) const
  {
    if(points.cols()!=3 || closest.cols()!=3 || points.rows()==0 ||
       closest.rows()!=points.rows() || indices.size()!=points.rows() ||
       !points.allFinite() || !closest.allFinite() ||
       indices.minCoeff()<0 || indices.maxCoeff()>=faces.rows())
      throw std::invalid_argument("Expected matching finite queries and cached closest features");
    if(!std::isfinite(minimum_double_area) || minimum_double_area<0 ||
       !std::isfinite(epsilon) || epsilon<0)
      throw std::invalid_argument("Tolerances must be finite and nonnegative");
    Eigen::VectorXf signs(points.rows());
    FloatMatrix normals(points.rows(),3);
    const nb::DRef<const FloatMatrix> vertex_ref(vertices);
    const nb::DRef<const Eigen::MatrixXI> face_ref(faces);
    igl::parallel_for(points.rows(),[&](const int p)
    {
      Eigen::RowVector3f q=points.row(p),c=closest.row(p),n;
      float sign;
      igl::pseudonormal_test_with_tolerances(
        vertex_ref,face_ref,face_normals,vertex_normals,edge_normals,edge_map,
        q,int(indices(p)),c,sign,n,minimum_double_area,epsilon);
      signs(p)=sign;
      normals.row(p)=n;
    },10000);
    return std::make_tuple(signs,normals);
  }
};

NB_MODULE(_tolerance_sweep,m)
{
  nb::class_<PreparedMesh>(m,"PreparedMesh")
    .def(nb::init<const nb::DRef<const FloatMatrix> &,
      const nb::DRef<const Eigen::MatrixXI> &>(),"vertices"_a.noconvert(),"faces"_a)
    .def("classify",&PreparedMesh::classify,"points"_a.noconvert(),"indices"_a,
      "closest"_a.noconvert(),"minimum_double_area"_a,"epsilon"_a);
}
