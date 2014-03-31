#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>

#include <vector>

#include <deal.II/grid/grid_out.h>

#include <cmath>
#include <algorithm>
#include "../../../../dealII/include/deal.II/lac/sparse_matrix.h"
#include "../../../../dealII/include/deal.II/bundled/boost/graph/stoer_wagner_min_cut.hpp"
#include "../../../../dealII/include/deal.II/grid/tria.h"

using namespace std;
using namespace dealii;

template<int dim, int quad>
class MatrixStudy {
public:
  MatrixStudy() ;
  void MakeMatrixes() ;
  
private:
  void make_grid() ;
  void setup_system() ;
  void assemble_system() ;
  
  
  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;
  
  
  SparseMatrix<double>  dd_matrix;
  SparseMatrix<double> df_matrix;
  SparseMatrix<double> ff_matrix;
  
  SparsityPattern sparsity_pattern;
  
};

