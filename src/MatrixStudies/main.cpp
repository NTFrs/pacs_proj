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
#include <deal.II/lac/sparse_direct.h>


#include <cmath>
#include <algorithm>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// #include "../../../../dealII/include/deal.II/lac/sparse_matrix.h"
// #include "../../../../dealII/include/deal.II/bundled/boost/graph/stoer_wagner_min_cut.hpp"
// #include "../../../../dealII/include/deal.II/grid/tria.h"

using namespace std;
using namespace dealii;

template<int dim>
class Boundary_Left_Side : public Function<dim>
{
public:
	Boundary_Left_Side() : Function< dim>() {};

	virtual double value (const Point<dim> &p, const unsigned int component =0) const;

};


template<int dim>
double Boundary_Left_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return 0;

}


template<int dim>
class Boundary_Right_Side: public Function<dim>
{
public:
	Boundary_Right_Side(double K) : Function< dim>(), _K(K){};
	virtual double value (const Point<dim> &p, const unsigned int component =0) const;
private:
	double _K;
};


template<int dim>
double Boundary_Right_Side<dim>::value(const Point<dim> &p, const unsigned int component) const
{
	Assert (component == 0, ExcInternalError());
	return _K;

}



template<int dim, int quad>
class MatrixStudy {
public:
	MatrixStudy() ;
	
	//calls all methods to build matrixes and prints them
	void print_matrixes(ostream& out);

	void study_conditions(ostream& out,  double K);
	
	void make_all();
	
private:
	//makes the grid,  a hypercube
	void make_grid() ;
	//sets up the system and matrix pattern
	void setup_system() ;
	//assembles the system through quadrature
	void assemble_system();
	
	void print_state(ostream &out);



	Triangulation<dim> triangulation;
	FE_Q<dim> fe;
	DoFHandler<dim> dof_handler;

	// sparsity_pattern be declared before matrixes! Cause matrixes depend
	// on it and destructor gets angry if you try to destroy him before matrixes
	// (it's counter is different form zero)
	SparsityPattern sparsity_pattern;                          
	SparseMatrix<double>  dd_matrix;
	SparseMatrix<double> fd_matrix;
	SparseMatrix<double> ff_matrix;
	SparseMatrix<double> system_matrix;
	SparseMatrix<double> m2_matrix;
	
	Vector<double>       solution;
	Vector<double>       system_rhs;


	

  };

template<int dim, int quad>
MatrixStudy<dim,quad>::MatrixStudy() : fe(1), dof_handler(triangulation) {}

template<int dim, int quad>
void MatrixStudy<dim,quad>::make_grid() {

	//simple mesh generation
	GridGenerator::hyper_cube(triangulation,0,0.8);
	triangulation.refine_global(3);

	std::cout << "   Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl
	<< "   Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl;
  }


template<int dim, int quad>
void MatrixStudy<dim,quad>::setup_system() {

	dof_handler.distribute_dofs(fe);

	std::cout << "   Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< std::endl;

	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);

	sparsity_pattern.copy_from(c_sparsity);

	std::ofstream out ("sparsity_pattern1.aux");
	sparsity_pattern.print_gnuplot (out);

	dd_matrix.reinit(sparsity_pattern);
	fd_matrix.reinit(sparsity_pattern);
	ff_matrix.reinit(sparsity_pattern);	
	system_matrix.reinit(sparsity_pattern);	
	m2_matrix.reinit(sparsity_pattern);
	
	system_rhs.reinit(dof_handler.n_dofs());

	solution.reinit(dof_handler.n_dofs());

  }

template<int dim,  int quad>
void MatrixStudy<dim, quad>::assemble_system() {

	QGauss<dim> quadrature_formula(quad);
	FEValues<dim> fe_values (fe, quadrature_formula, update_values   | update_gradients |
	 update_JxW_values);

	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();

	cout<< "Assembling System\n";
	cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
	cout<< "Quadrature points per cell: "<< n_q_points<< endl;

	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

	FullMatrix<double> cell_dd(dofs_per_cell);
	FullMatrix<double> cell_fd(dofs_per_cell);
	FullMatrix<double> cell_ff(dofs_per_cell);

	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();
	Tensor< 1 , dim, double > ones(1.);
	
// 	Tensor< 1 , dim, double > increasing;
	cout<< "Tensore ones prima"<< ones<< endl;

	for (unsigned i=0;i<dim;++i) {
	 ones[i]=1;
// 	 increasing[i]=i;
   }

	cout << "Tensore ones dopo" << ones << endl;
// 	cout << "Prodotto tensori " << ones*increasing<< endl;

	for (; cell !=endc;++cell) {
	 fe_values.reinit(cell);
	 cell_dd=0;
	 cell_fd=0;
	 cell_ff=0;
	 for (unsigned q_point=0;q_point<n_q_points;++q_point)
	 for (unsigned i=0;i<dofs_per_cell;++i)
	 for (unsigned j=0; j<dofs_per_cell;++j) {
/*
	  cout<<fe_values.JxW(q_point)<< " è il JxW quà\n";
	  cout<< fe_values.shape_grad(i, q_point)<< " e "<< fe_values.shape_grad(j, q_point)<< " gradiente\n";
	  cout<< fe_values.shape_value(i, q_point)<< " e " << fe_values.shape_value(j, q_point)<< " funzione\n";
		*/
		//a lot of time lost on this: it's important to summ all q_points (obvious,  but we forgot to use +=)
		cell_dd(i, j)+=fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point)*fe_values.JxW(q_point);
		cell_fd(i, j)+=fe_values.shape_value(i, q_point)*(ones*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);
		cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);

	}

	 cell->get_dof_indices (local_dof_indices);

	 for (unsigned int i=0; i<dofs_per_cell;++i)
	 for (unsigned int j=0; j< dofs_per_cell; ++j) {

	  dd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_dd(i, j));
	  fd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_fd(i, j));
	  ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));

	}

   }


  }

  
  
template<int dim,  int quad>
void MatrixStudy<dim, quad>::make_all() {
	
	make_grid();
	setup_system();
	assemble_system();
	
}
  
template<int dim,  int quad>
void MatrixStudy<dim, quad>::print_matrixes(ostream& out)
{



	out<<"Derivata Derivata:\n";
	dd_matrix.print_formatted(out);

	out<<"Funzione Derivata:\n";
	fd_matrix.print_formatted(out);

	out<<"Funzone Funzione:\n";
	ff_matrix.print_formatted(out);

}

template<int dim,  int quad>
void MatrixStudy<dim, quad>::print_state(ostream &out) {
	
	out<< "System Matrix:\n";
	system_matrix.print_formatted(out);
	out<< "System M2:\n";
	m2_matrix.print_formatted(out);
	out<< "System solution:\n";
	solution.print(out);
	out<< "System rhs:\n";
	system_rhs.print(out);
	
}

template<int dim,  int quad>
void MatrixStudy<dim, quad>::study_conditions(ostream& out,  double K) {


	double diff(1),  trasp(2),  reaz(12);
	
	system_matrix.add(reaz, ff_matrix);
	system_matrix.add(-trasp, fd_matrix);
	system_matrix.add(diff, dd_matrix);
	
	m2_matrix.add(6, ff_matrix);
	solution.add(1.);
	m2_matrix.vmult(system_rhs, solution);
	
	print_state(out);
	
	{
	 std::map<types::global_dof_index,double> boundary_values;
	 VectorTools::interpolate_boundary_values (dof_handler,
	  0,
	  Boundary_Left_Side<dim>(),
	  boundary_values);


	 VectorTools::interpolate_boundary_values (dof_handler,
	  1,
	  Boundary_Right_Side<dim>(K),
	  boundary_values);

	 MatrixTools::apply_boundary_values (boundary_values,
	  system_matrix,
	  solution,
	  system_rhs);
	}
	
	cout << "#########################################################CONDITIONS##########################################################\n";
	
	print_state(out);
	Vector<double> old_rhs(system_rhs);
	
	SparseDirectUMFPACK solver;
	solver.initialize(sparsity_pattern);
	solver.factorize(system_matrix);
	solver.solve(system_rhs);
	solution=system_rhs;
	cout << "##############################################################SOLVED##########################################################\n";
	
	print_state(out);
	cout<< "old_rhs is :\n";
	old_rhs.print();
}



template<int T>
void print_tensor_product() {

	Tensor< 1 , T, double > ones;
	Tensor< 1 , T, double > increasing;

	for (unsigned int i=0;i<T;++i) {
	 ones[i]=1;
	 increasing[i]=i+1;
   }

	cout << "Tensore ones " << ones << endl;
	cout << "Tensore increasing " << increasing << endl;
	cout << "Prodotto tensori " << ones*increasing<< endl;

  }

int main () {
/*
	print_tensor_product<1>();
	print_tensor_product<2>();
	print_tensor_product<3>();
 */

// 	MatrixStudy<1, 1> M1;
// 	M1.print_matrixes(cout);
	MatrixStudy<1, 2> M2;
	M2.make_all();
	M2.print_matrixes(cout);
	M2.study_conditions(cout, 10);
 	
// 	MatrixStudy<1, 3> M3;
// 	M3.print_matrixes(cout);

	
	/*
	MatrixStudy<2, 1> M21;
	MatrixStudy<2, 2> M22;
	M21.print_matrixes(cout);
	M22.print_matrixes(cout);
	*/


	return 0;
  }
















