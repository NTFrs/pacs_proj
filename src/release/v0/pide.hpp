#include "deal_ii.hpp"

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

#include "models.hpp"

using namespace dealii;

template<unsigned dim>
class Option
{
private:
        // Model and Option parameters
        std::vector<BlackScholesModel>       models;
        double                  r;
        double                  T;
        
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
        unsigned                N; // space
        unsigned                M; // time
        double                  price;
	bool                    ran;
        
        // We add the model classes as friend in order to easily get access to the parameters
        friend class            BlackScholesModel;
        
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
               unsigned N_,
               unsigned M_);
        // Cosntructor 2d
        Option(BlackScholesModel const &model1,
               BlackScholesModel const &model2,
               double r_,
               double T_,
               unsigned N_,
               unsigned M_);
        
        void run()
        {
                make_grid();
                setup_system();
                assemble_system();
                solve();
        };
        
        inline double get_price() const { return price; };
};

// Constructor 1d
template <unsigned dim>
Option<dim>::Option(BlackScholesModel const &model,
                    double r_,
                    double T_,
                    unsigned N_,
                    unsigned M_)
{
        // error!
}

// Constructor 1d specialized
template <>
Option<1>::Option(BlackScholesModel const &model,
                  double r_,
                  double T_,
                  unsigned N_,
                  unsigned M_)
:
r(r_),
T(T_),
N(N_),
M(M_),
fe (1),
dof_handler (triangulation)
{
        models.emplace_back(model);
}

// Constructor 2d
template <unsigned dim>
Option<dim>::Option(BlackScholesModel const &model1,
                    BlackScholesModel const &model2,
                    double r_,
                    double T_,
                    unsigned N_,
                    unsigned M_)
{
        // error!
}

// Constructor 2d specialized
template <>
Option<2>::Option(BlackScholesModel const &model1,
                  BlackScholesModel const &model2,
                  double r_,
                  double T_,
                  unsigned N_,
                  unsigned M_)
:
r(r_),
T(T_),
N(N_),
M(M_),
fe (1),
dof_handler (triangulation)
{
        // check model1==model2
        models.emplace_back(model1);
        models.emplace_back(model2);
}

// make_grid
template<unsigned dim>
void Option<dim>::make_grid(){
        // error
}

// make_grid 1d
template <>
void Option<1>::make_grid(){
        // error
}


