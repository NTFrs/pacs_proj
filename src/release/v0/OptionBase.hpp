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
#include <exception>

#include "boundary_conditions.hpp"
#include "BoundaryConditions.hpp"
#include "FinalConditions.hpp"
#include "OptionTypes.hpp"
#include "models.hpp"
#include "constants.hpp"
#include "Densities.hpp"
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
        
        enum class ModelType
        {
                BlackScholes,
                Merton,
                Kou
        };
        
        ModelType               model_type;
        
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
        double                  f;
	bool                    ran;
        
        // Integral Part
        LevyIntegral<dim> *     levy;       
        
        // Private methods
        virtual void make_grid();
        virtual void setup_system();
        virtual void assemble_system();
        // solve method equal to 0, in order to make this class abstract
        // we choose this method because it's very "option"-dependent
        virtual void solve() = 0;
        
public:
        //! Constructor 1d
        /*!
         * Constructor 1d called by inherited classes.
         */
        OptionBase(ExerciseType type_,
                   Model * const model,
                   double r_,
                   double T_,
                   double K_,
                   unsigned refs_,
                   unsigned time_step_);
        
        //! Cosntructor 2d
        /*!
         * Constructor 2d called by inherited classes.
         */
        OptionBase(ExerciseType type_,
                   Model * const model1,
                   Model * const model2,
                   double rho_,
                   double r_,
                   double T_,
                   double K_,
                   unsigned refs_,
                   unsigned time_step_);
        
        //! Destructor
        virtual ~OptionBase(){
                delete levy;
                delete system_matrix;
        };
        
        //! 
        /*!
         * This function creates the system and solves it.
         */
        virtual void run()
        {
                make_grid();
                setup_system();
                assemble_system();
                solve();
        };
        
        //! SISTEMAREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE
        /*!
         * This function is used to set the scale factor of the grid boundary.
         * \param f_    is set by default to 0.5. The user can specify it in ]0,1[.
         */
        virtual void set_scale_factor(double f_) {
                if (f_<=0. && f_>=1.) {
                        throw(std::logic_error("Error! The scale factor must be in ]0,1[.\n"));
                }
                else
                        f=f_;
        };
        
        //!
        /*!
         * This function returns the price of the option
         */
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
        throw(std::logic_error("Error! Dimension must be 1 or 2.\n"));
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
f(0.5),
ran(false),
levy(NULL)
{
        models.push_back(model);
        
        BlackScholesModel       *     bs(dynamic_cast<BlackScholesModel *> (model));
        KouModel                *     kou(dynamic_cast<KouModel *> (model));
        MertonModel             *     mer(dynamic_cast<MertonModel *> (model));
        
        if (bs) 
                model_type=ModelType::BlackScholes;
        
        else if (kou) {
                model_type=ModelType::Kou;
        }
        
        else if (mer)
                model_type=ModelType::Merton;
        
        else    
                throw(std::logic_error("Error! Unknown models.\n"));
        
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
        throw(std::logic_error("Error! Dimension must be 1 or 2.\n"));
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
f(0.5),
ran(false),
levy(NULL)
{
        models.push_back(model1);
        models.push_back(model2);
        
        BlackScholesModel       *     bs(dynamic_cast<BlackScholesModel *> (model1));
        KouModel                *     kou(dynamic_cast<KouModel *> (model1));
        MertonModel             *     mer(dynamic_cast<MertonModel *> (model1));
        
        if (bs) {
                model_type=ModelType::BlackScholes;
                BlackScholesModel * bs2(dynamic_cast<BlackScholesModel *> (model2));
                if (!bs2)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else if (kou) { 
                model_type=ModelType::Kou;
                KouModel * kou2(dynamic_cast<KouModel *> (model2));
                if (!kou2)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else if (mer) {
                model_type=ModelType::Merton;
                MertonModel * mer2(dynamic_cast<MertonModel *> (model2));
                if (!mer2)
                        throw(std::logic_error("Error! Different types of model.\n"));
        }
        else    
                throw(std::logic_error("Error! Unknown models.\n"));
        
}

// make grid
template<unsigned dim>
void OptionBase<dim>::make_grid(){
        
        std::vector<unsigned> refinement(dim);
        
        for (unsigned i=0; i<dim; ++i) {
                
                Smin[i]=(1-f)*(*models[i]).get_spot()*
                exp((r-(*models[i]).get_vol()*(*models[i]).get_vol()/2)*T
                    -(*models[i]).get_vol()*sqrt(T)*6);
                
                Smax[i]=(1+f)*(*models[i]).get_spot()*
                exp((r-(*models[i]).get_vol()*(*models[i]).get_vol()/2)*T
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
        
        if (model_type==ModelType::Kou) {
                
                if (dim==1) {
                        levy=new KouIntegral<dim>(dynamic_cast<KouModel *> (models[0]->get_pointer()),
                                                  Smin, Smax);
                }
                
        }
        
        else if (model_type==ModelType::Merton) {
                
                Function<1> * m=new Merton_Density<1>();
                
                if (dim==1) {
                        levy=new LevyIntegral<dim>(m, Smin, Smax);
                }
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
        double alpha(0.);
        double lambda(0.);
        
        if (model_type!=ModelType::BlackScholes && dim==1) {
                alpha=levy->get_part1();
                lambda=models[0]->get_lambda();
        }
        
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
                                trasp[0]=(r-(*models[0]).get_vol()*(*models[0]).get_vol()-alpha)*quad_points[q_point][0];
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
                                        
                                        
                                        cell_mat(i, j)+=fe_values.JxW(q_point)*
                                        (
                                         (1/dt+r+lambda)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
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