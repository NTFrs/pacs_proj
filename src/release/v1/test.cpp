#include <iostream>
// #include "Levy.hpp"
#include "LevyIntegralPrice.hpp"
#include "LevyIntegralLogPrice.hpp"
#include <vector>
#include "models.hpp"
#include "BoundaryConditions.hpp"
#include "OptionTypes.hpp"


/// ////////////////////////////////////////////////////////
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
// And this is the file in which the functions are declared that create grids:
#include <deal.II/grid/grid_generator.h>

// The next three files contain classes which are needed for loops over all
// cells and to get the information from the cell objects. The first two have
// been used before to get geometric information from cells; the last one is
// new and provides information about the degrees of freedom local to a cell:
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>

// In this file contains the description of the Lagrange interpolation finite
// element:
#include <deal.II/fe/fe_q.h>

// And this file is needed for the creation of sparsity patterns of sparse
// matrices, as shown in previous examples:
#include <deal.II/dofs/dof_tools.h>

// The next two file are needed for assembling the matrix using quadrature on
// each cell. The classes declared in them will be explained below:
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

// The following three include files we need for the treatment of boundary
// values:
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

// We're now almost to the end. The second to last group of include files is
// for the linear algebra which we employ to solve the system of equations
// arising from the finite element discretization of the Laplace equation. We
// will use vectors and full matrices for assembling the system of equations
// locally on each cell, and transfer the results into a sparse matrix. We
// will then use a Conjugate Gradient solver to solve the problem, for which
// we need a preconditioner (in this program, we use the identity
// preconditioner which does nothing, but we need to include the file anyway):
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

// Finally, this is for output to a file and to the console:
#include <deal.II/numerics/data_out.h>
#include <boost/graph/stoer_wagner_min_cut.hpp>
#include <fstream>
#include <iostream>

/// /////////////////////////////////////////////////////////////////







using namespace std;

int main(){
	
		using namespace dealii;
		
		KouModel Mod1(100,0.2, 0.5, 3, 2, 4);
		KouModel Mod2(120, 0.25, 0.4, 4, 1, 5);

		/*
		vector<Model *> ModPtr2;
		ModPtr2.push_back(&Mod1);
		LevyIntegralPrice<1> Int2(dealii::Point<1>(0.),dealii::Point<1>(100.), ModPtr2);
		*/
 
		vector<Model *> ModPtr;
		ModPtr.push_back(&Mod1);
		ModPtr.push_back(&Mod2);
/*		
		LevyIntegralPrice<2> Int(dealii::Point<2>(0.5, 0.5),dealii::Point<2>(100.,100.), ModPtr);
		KouIntegralPrice<2> Kou(ModPtr);
*/		
		BoundaryCondition<2> BC(100, 1, 0.03, OptionType::Call);
		LevyIntegralLogPrice<2> Int(dealii::Point<2>(-2., -2.),dealii::Point<2>(2.,2.), ModPtr, &BC);
		Triangulation<2> triangulation;

		GridGenerator::hyper_cube (triangulation);
		triangulation.refine_global (4);
		FE_Q<2>              fe(1);
		DoFHandler<2>        dof_handler(triangulation);

		dof_handler.distribute_dofs(fe);
		Vector<double> solution;
		solution.reinit(dof_handler.n_dofs());
		for (unsigned i=0;i<solution.size();++i)
		  solution[i]=i;
		
		Int.compute_J(solution, dof_handler, fe);
		
		
		
/* 
        BlackScholesModel model(95., 0.120381);
        
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.6599, 3.13868);
        
        EuropeanOptionPrice<1> foo(OptionType::Call, model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionLogPrice<1> foo2
        (OptionType::Call, model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionPrice<1> goofy(OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        AmericanOptionPrice<1> minnie(model.get_pointer(), 0.0367, 1., 90., 10, 100);
        
        EuropeanOptionPrice<2> mickey(OptionType::Call, model1.get_pointer(), model2.get_pointer(),
                                 -0.2, 0.1, 1., 200., 7, 100);
        
        EuropeanOptionPrice<1> duffy(OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
        
        foo.run();
        foo2.run();
        goofy.run();
        minnie.run();
        mickey.run();
        duffy.run();
        
        cout<<foo.get_price()<<"\n";
        cout<<foo2.get_price()<<"\n";
        cout<<goofy.get_price()<<"\n";
        cout<<minnie.get_price()<<"\n";
        cout<<mickey.get_price()<<"\n";
        cout<<duffy.get_price()<<"\n";
        
        
        cout<<"TARGET (Premia)\n"
            <<"PDE  1d Call  (10, 100): 9.62216.\n"
            <<"PDE  1d Put   (10, 100): 1.43609.\n"
            <<"PDE  1d PutAm (10, 100): 1.56720.\n"
            <<"PDE  2d Call  (7,  100): 21.3480.\n"
            <<"PIDE 1d Call  (8,  100): 12.3683.\n";
*/        
        return 0;
        
}