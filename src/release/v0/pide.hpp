#include "deal_ii.hpp"
#include "matrix_with_psor.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <string>

#include "boundary_conditions.hpp"
#include "models.hpp"
#include "constants.hpp"

using namespace dealii;
using namespace std;

template<unsigned dim>
class Option
{
private:
        // Model and Option parameters
        std::vector<BlackScholesModel>       models;
        double                  r;
        double                  T;
        double                  K;
        double                  rho;
        
        // Triangulation and fe objects
        Triangulation<dim>      triangulation;
	FE_Q<dim>               fe;
	DoFHandler<dim>         dof_handler;
        
        // Matrices
        SparsityPattern         sparsity_pattern;
	SparseMatrix<double>    system_matrix;
	SparseMatrix<double>    system_M2;
	SparseMatrix<double>    dd_matrix;
	SparseMatrix<double>    fd_matrix;
	SparseMatrix<double>    ff_matrix;
	
        // Solution and rhs vectors
	Vector<double>          solution;
	Vector<double>          system_rhs;
        
        // Mesh boundaries
        Point<dim>              Smin, Smax;

        // Disctretization parameters
        unsigned                refs;      // space
        unsigned                time_step; // time
        double                  dt;
        double                  price;
	bool                    ran;
        
        // Private methods
        void make_grid();
        void setup_system();
        void assemble_system();
        void solve();
        
public:
        // Constructor 1d
        Option(BlackScholesModel const &model,
               double r_,
               double T_,
               double K_,
               unsigned refs_,
               unsigned time_step_);
        // Cosntructor 2d
        Option(BlackScholesModel const &model1,
               BlackScholesModel const &model2,
               double r_,
               double T_,
               double K_,
               unsigned refs_,
               unsigned time_step_);
        
        // Destructor
        // virtual ~Option(){};
        
        void run()
        {
                make_grid();
                setup_system();
                assemble_system();
                solve();
        };
        
        inline double get_price();
};

// Constructor 1d
template <unsigned dim>
Option<dim>::Option(BlackScholesModel const &model,
                    double r_,
                    double T_,
                    double K_,
                    unsigned refs_,
                    unsigned time_step_)
{
        // error!
}

// Constructor 1d specialized
template <>
Option<1>::Option(BlackScholesModel const &model,
                  double r_,
                  double T_,
                  double K_,
                  unsigned refs_,
                  unsigned time_step_)
:
r(r_),
T(T_),
K(K_),
fe (1),
dof_handler (triangulation),
refs(refs_),
time_step(time_step_),
dt(T/static_cast<double>(time_step_)),
price(0.),
ran(false)
{
        models.emplace_back(model);
}

// Constructor 2d
template <unsigned dim>
Option<dim>::Option(BlackScholesModel const &model1,
                    BlackScholesModel const &model2,
                    double r_,
                    double T_,
                    double K_,
                    unsigned refs_,
                    unsigned time_step_)
{
        // error!
}

// Constructor 2d specialized
template <>
Option<2>::Option(BlackScholesModel const &model1,
                  BlackScholesModel const &model2,
                  double r_,
                  double T_,
                  double K_,
                  unsigned refs_,
                  unsigned time_step_)
:
r(r_),
T(T_),
K(K_),
fe (1),
dof_handler (triangulation),
refs(refs_),
time_step(time_step_),
dt(T/static_cast<double>(time_step_)),
price(0.),
ran(false)
{
        // check model1==model2
        models.emplace_back(model1);
        models.emplace_back(model2);
}

// make grid
template<unsigned dim>
void Option<dim>::make_grid(){
        
        std::vector<unsigned> refinement(dim);
        
        for (unsigned i=0; i<dim; ++i) {
                Smin[i]=models[i].get_spot()*exp((r-models[i].get_vol()*models[i].get_vol()/2)*T
                                                 -models[i].get_vol()*sqrt(T)*6);
                Smin[i]=models[i].get_spot()*exp((r-models[i].get_vol()*models[i].get_vol()/2)*T
                                                 +models[i].get_vol()*sqrt(T)*6);
                refinement[i]=pow(2, refs);
        }
        
        GridGenerator::subdivided_hyper_rectangle (triangulation, refinement, Smin, Smax);
        
        return;
}

// setup system
template<unsigned dim>
void Option<dim>::setup_system()
{
        
	dof_handler.distribute_dofs(fe);
        
	std::cout << "   Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< std::endl;
        
	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
        
	sparsity_pattern.copy_from(c_sparsity);
        
        /*
         if (sono americana)
                system_matrix=SparseMatrix_withProjectedSOR(...);
         */
        
        dd_matrix.reinit(sparsity_pattern);
	fd_matrix.reinit(sparsity_pattern);
	ff_matrix.reinit(sparsity_pattern);
	system_matrix.reinit(sparsity_pattern);
	system_M2.reinit(sparsity_pattern);
        
	typename Triangulation<dim>::cell_iterator
	cell = triangulation.begin (),
	endc = triangulation.end();
        
        if (dim==1) {
        
                for (; cell!=endc; ++cell)
                        for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                                if (cell->face(face)->at_boundary())
                                        if (std::fabs(cell->face(face)->center()(0) - (Smax[0])) < 
                                            constants::light_toll)
                                                cell->face(face)->set_boundary_indicator (1);
                
        }
        
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
        
        return;
        
}

// assemble system
template<unsigned dim>
void Option<dim>::assemble_system()
{
        QGauss<dim> quadrature_formula(2*dim);
	FEValues<dim> fe_values (fe, quadrature_formula, update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points);
        
	const unsigned int   dofs_per_cell = fe.dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
        
	cout<< "Assembling System\n";
	cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
	cout<< "Quadrature points per cell: "<< n_q_points<< endl;
        
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
       
	FullMatrix<double> cell_ff(dofs_per_cell);
	FullMatrix<double> cell_mat(dofs_per_cell);
        
	typename DoFHandler<dim>::active_cell_iterator
	cell=dof_handler.begin_active(),
	endc=dof_handler.end();
	Tensor< 1 , dim, double > trasp;
	Tensor< 2, dim,  double > sig_mat;
	
	vector<Point<dim> > quad_points(n_q_points);
        
        for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                cell_ff=0;
                cell_mat=0;
                
                quad_points=fe_values.get_quadrature_points();
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                        
                        if (dim==1) {
                                trasp[0]=(r-models[0].get_vol()*models[0].get_vol())*quad_points[q_point][0];
                                sig_mat[0][0]=0.5*models[0].get_vol()*models[0].get_vol()
                                *quad_points[q_point][0]*quad_points[q_point][0];
                        }
                        
                        else if (dim==2) {
                                
                                trasp[0]=models[0].get_vol()*models[0].get_vol()*quad_points[q_point][0]
                                +0.5*rho*models[0].get_vol()*models[1].get_vol()*quad_points[q_point][0]
                                -r*quad_points[q_point][0];
                                
                                trasp[1]=models[1].get_vol()*models[1].get_vol()*quad_points[q_point][1]
                                +0.5*rho*models[0].get_vol()*models[1].get_vol()*quad_points[q_point][1]
                                -r*quad_points[q_point][1];
                                
                                sig_mat[0][0]=0.5*models[0].get_vol()*models[0].get_vol()
                                *quad_points[q_point][0]*quad_points[q_point][0];
                                sig_mat[1][1]=0.5*models[1].get_vol()*models[1].get_vol()
                                *quad_points[q_point][1]*quad_points[q_point][1];
                                sig_mat[0][1]=0.5*rho*models[0].get_vol()*models[1].get_vol()*
                                quad_points[q_point][0]*quad_points[q_point][1];
                                sig_mat[1][0]=sig_mat[0][1];
                        }
                        
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        cell_mat(i, j)+=fe_values.JxW(q_point)*
                                        (
                                         (1/dt+r)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
                                         +fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
                                         +fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point) //segno + o - ??
                                         );
                                        
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                }
                }
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell;++i)
                        for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                
                                system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_mat(i, j));
                                ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                
                        }
        }
        
	system_M2.add(1/dt, ff_matrix);
        
        return;
        
}

template<unsigned dim>
void Option<dim>::solve() {
        
	VectorTools::interpolate (dof_handler, PayOff<dim>(K), solution);
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (dof_handler);
                data_out.add_data_vector (solution, "begin");
                
                data_out.build_patches ();
                
                std::ofstream output ("begin.gpl");
                data_out.write_gnuplot (output);
        }
        
	unsigned int Step=time_step;
        
	Boundary_Right_Side<dim> right_bound(K, T, r);
	cout<< "time step is"<< time_step<< endl;
	
	for (double time=T-dt;time >=0;time-=dt, --Step) {
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                system_M2.vmult(system_rhs, solution);
                right_bound.set_time(time);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  0,
                                                                  Boundary_Left_Side<dim>(),
                                                                  boundary_values);
                        
                        
                        VectorTools::interpolate_boundary_values (dof_handler,
                                                                  1,
                                                                  right_bound,
                                                                  boundary_values);
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            system_matrix,
                                                            solution,
                                                            system_rhs, false);
                        
                }
                
                SparseDirectUMFPACK solver;
                solver.initialize(sparsity_pattern);
                solver.factorize(system_matrix);
                solver.solve(system_rhs);
                
                solution=system_rhs;
                
        }
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (dof_handler);
                data_out.add_data_vector (solution, "end");
                
                data_out.build_patches ();
                
                std::ofstream output ("end.gpl");
                data_out.write_gnuplot (output);
        }
        
	ran=true;

}

template<unsigned dim>
double Option<dim>::get_price() {
        
	if (ran==false) {
                this->run();
        }
        
        Point<dim> p(models[0].get_spot());
	Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
	return fe_function.value(p);
}

