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
#include "OptionParameters.hpp"

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
        virtual void make_grid() = 0;
        virtual void setup_system();
        // Pure abstract methods
        virtual void assemble_system() = 0;
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
        virtual inline double get_price()=0;
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

#endif