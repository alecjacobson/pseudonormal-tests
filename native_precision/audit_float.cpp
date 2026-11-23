// Compile with -Werror=double-promotion to audit the pseudonormal query helpers.
#include <igl/AABB.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/per_edge_normals.h>
#include <igl/pseudonormal_test.h>

void audit_float(const Eigen::MatrixXf &vertices, const Eigen::MatrixXi &faces,
                 const Eigen::RowVector3f &query)
{
  Eigen::MatrixXf face_normals,vertex_normals,edge_normals;
  Eigen::MatrixXi edges;
  Eigen::VectorXi edge_map;
  igl::per_face_normals(vertices,faces,face_normals);
  igl::per_vertex_normals(vertices,faces,
    igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,face_normals,vertex_normals);
  igl::per_edge_normals(vertices,faces,
    igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,face_normals,edge_normals,edges,edge_map);
  igl::AABB<Eigen::MatrixXf,3> tree;
  tree.init(vertices,faces);
  int closest_face;
  Eigen::RowVector3f closest,normal;
  tree.squared_distance(vertices,faces,query,closest_face,closest);
  float sign;
  igl::pseudonormal_test(vertices,faces,face_normals,vertex_normals,edge_normals,
    edge_map,query,closest_face,closest,sign,normal);
}
