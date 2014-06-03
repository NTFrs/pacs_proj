#ifndef __option_base_hpp
#define __option_base_hpp

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
#include <memory>

#include "boundary_conditions.hpp"
#include "BoundaryConditions.hpp"
#include "FinalConditions.hpp"
#include "OptionTypes.hpp"
#include "models.hpp"
#include "constants.hpp"
#include "LevyIntegral.hpp"

using namespace dealii;
using namespace std;

template<unsigned dim>
class OptionBase
{
protected:
        // Model and Option parameters
        ExerciseType            type;
        std::vector<Model *>    models;
        double                  rho;
        double                  r;
        double                  T;
        double                  K;
        
        // Triangulation and fe objects
        Triangulation<dim>      triangulation;
	FE_Q<dim>               fe;
	DoFHandler<dim>         dof_handler;
        
        // Matrices
        SparsityPattern         sparsity_pattern;
	
        SparseMatrix_withProjectedSOR<double, dim> * matrix_with_sor;
        SparseMatrix<double> * system_matrix;
	
        SparseMatrix<double>    system_M2;
	SparseMatrix<double>    dd_matrix;
	SparseMatrix<double>    fd_matrix;
	SparseMatrix<double>    ff_matrix;
        
        // points of grid
        std::vector< Point<dim> >       grid_points;
	
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
        virtual void make_grid();
        virtual void setup_system();
        virtual void assemble_system();
        // solve method equal to 0, in order to make this class abstract
        // we choose this method because it's very "option"-dependent
        virtual void solve() = 0;
        
public:
        // Constructor 1d
        OptionBase(ExerciseType type_,
                   Model * const model,
                   double r_,
                   double T_,
                   double K_,
                   unsigned refs_,
                   unsigned time_step_);
        
        // Cosntructor 2d
        OptionBase(ExerciseType type_,
                   Model * const model1,
                   Model * const model2,
                   double rho_,
                   double r_,
                   double T_,
                   double K_,
                   unsigned refs_,
                   unsigned time_step_);
        
        // Destructor
        virtual ~OptionBase(){
                delete system_matrix;
        };
        
        virtual void run()
        {
                make_grid();
                setup_system();
                assemble_system();
                solve();
        };
        
        virtual inline double get_price();
};

// Constructor 1d
template <unsigned dim>
OptionBase<dim>::OptionBase(ExerciseType type_,
                            Model * const model,
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
OptionBase<1>::OptionBase(ExerciseType type_,
                          Model * const model,
                          double r_,
                          double T_,
                          double K_,
                          unsigned refs_,
                          unsigned time_step_)
:
type(type_),
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
        models.push_back(model);
}

// Constructor 2d
template <unsigned dim>
OptionBase<dim>::OptionBase(ExerciseType type_,
                            Model * const model1,
                            Model * const model2,
                            double rho_,
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
OptionBase<2>::OptionBase(ExerciseType type_,
                          Model * const model1,
                          Model * const model2,
                          double rho_,
                          double r_,
                          double T_,
                          double K_,
                          unsigned refs_,
                          unsigned time_step_)
:
type(type_),
rho(rho_),
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
        models.push_back(model1);
        models.push_back(model2);
}

// make grid
template<unsigned dim>
void OptionBase<dim>::make_grid(){
        
        std::vector<unsigned> refinement(dim);
        
        for (unsigned i=0; i<dim; ++i) {
                Smin[i]=(*models[i]).get_spot()*exp((r-(*models[i]).get_vol()*(*models[i]).get_vol()/2)*T
                                                 -(*models[i]).get_vol()*sqrt(T)*6);
                Smax[i]=(*models[i]).get_spot()*exp((r-(*models[i]).get_vol()*(*models[i]).get_vol()/2)*T
                                                 +(*models[i]).get_vol()*sqrt(T)*6);
                refinement[i]=pow(2, refs);
        }
        
        GridGenerator::subdivided_hyper_rectangle (triangulation, refinement, Smin, Smax);
        
        grid_points=triangulation.get_vertices();
        
        return;
}

// setup system
template<unsigned dim>
void OptionBase<dim>::setup_system()
{
        
	dof_handler.distribute_dofs(fe);
        
	std::cout << "   Number of degrees of freedom: "
	<< dof_handler.n_dofs()
	<< std::endl;
        
	CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
        
	sparsity_pattern.copy_from(c_sparsity);
        
        if (type==ExerciseType::US) {
                matrix_with_sor=new SparseMatrix_withProjectedSOR<double, dim>;
                system_matrix=matrix_with_sor;
        }
        else {
                system_matrix=new SparseMatrix<double>;
                matrix_with_sor=NULL;
        }
        
        dd_matrix.reinit(sparsity_pattern);
	fd_matrix.reinit(sparsity_pattern);
	ff_matrix.reinit(sparsity_pattern);
	(*system_matrix).reinit(sparsity_pattern);
	system_M2.reinit(sparsity_pattern);
        
	solution.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());
        
        return;
        
}

// assemble system
template<unsigned dim>
void OptionBase<dim>::assemble_system()
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
	Tensor< 2 , dim, double > sig_mat;
	
	vector<Point<dim> > quad_points(n_q_points);
        
        for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                cell_ff=0;
                cell_mat=0;
                
                quad_points=fe_values.get_quadrature_points();
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                        
                        if (dim==1) {
                                trasp[0]=(r-(*models[0]).get_vol()*(*models[0]).get_vol())*quad_points[q_point][0];
                                sig_mat[0][0]=0.5*(*models[0]).get_vol()*(*models[0]).get_vol()
                                *quad_points[q_point][0]*quad_points[q_point][0];
                        }
                        
                        else if (dim==2) {
                                
                                trasp[0]=-((*models[0]).get_vol()*(*models[0]).get_vol()
                                *quad_points[q_point][0]
                                +0.5*rho*(*models[0]).get_vol()*(*models[1]).get_vol()
                                *quad_points[q_point][0]-r*quad_points[q_point][0]);
                                
                                trasp[1]=-((*models[1]).get_vol()*(*models[1]).get_vol()*
                                quad_points[q_point][1]+0.5*rho*(*models[0]).get_vol()*
                                (*models[1]).get_vol()*quad_points[q_point][1]
                                -r*quad_points[q_point][1]);
                                
                                sig_mat[0][0]=0.5*(*models[0]).get_vol()*(*models[0]).get_vol()
                                *quad_points[q_point][0]*quad_points[q_point][0];
                                sig_mat[1][1]=0.5*(*models[1]).get_vol()*(*models[1]).get_vol()
                                *quad_points[q_point][1]*quad_points[q_point][1];
                                sig_mat[0][1]=0.5*rho*(*models[0]).get_vol()*(*models[1]).get_vol()*
                                quad_points[q_point][0]*quad_points[q_point][1];
                                sig_mat[1][0]=sig_mat[0][1];
                        }
                        
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        if ( dim==1 )
                                                
                                                cell_mat(i, j)+=fe_values.JxW(q_point)*
                                                (
                                                 (1/dt+r)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
                                                 +fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
                                                 -fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point)
                                                 );
                                        
                                        else 
                                                cell_mat(i, j)+=fe_values.JxW(q_point)*
                                                (
                                                 (1/dt+r)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
                                                 +fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
                                                 -fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point)
                                                 );
                                        
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                }
                }
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell;++i)
                        for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                
                                (*system_matrix).add(local_dof_indices[i], local_dof_indices[j], cell_mat(i, j));
                                ff_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                
                        }
        }
        
	system_M2.add(1/dt, ff_matrix);
        
        return;
        
}

template<unsigned dim>
double OptionBase<dim>::get_price() {
        
	if (ran==false) {
                this->run();
        }
        
        Point<dim> p;
        
        for (unsigned i=0; i<dim; ++i) {
                p(i)=(*models[i]).get_spot();
        }
        
	Functions::FEFieldFunction<dim> fe_function (dof_handler, solution);
	return fe_function.value(p);
}

#endif