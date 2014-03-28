#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/grid_out.h>

#include <fstream>
#include <iostream>

using namespace std;
using namespace dealii;

const int dim=1;

/*////////////////////////////////////
  //////////// CLASSES ///////////////
  ////////////////////////////////////
 */

class HeatEquation
{
public:
        HeatEquation(double x);
        void run();
        
private:
        void setup_system();
        void solve_time_step();
        void output_results() const {};
        void build_matrix();
        //void refine_mesh (const unsigned int min_grid_level,
        //                  const unsigned int max_grid_level){};
        
        Triangulation<dim>   triangulation;
        FE_Q<dim>            fe;
        DoFHandler<dim>      dof_handler;
        
        //ConstraintMatrix     constraints;
        
        SparsityPattern      sparsity_pattern;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> laplace_matrix;
        SparseMatrix<double> system_matrix;
        
        Vector<double>       solution;
        Vector<double>       old_solution;
        Vector<double>       system_rhs;
        
        double               time;
        double               time_step;
        double               time_end;
        unsigned int         timestep_number;
        
        const double         theta;
};

class RightHandSide : public Function<dim>
{
public:
        RightHandSide ()
        :
        Function<dim>(),
        period (0.2)
        {}
        
        virtual double value (const Point<dim> &p,
                              const unsigned int component = 0) const;
        
private:
        const double period;
};

class BoundaryValues : public Function<dim>
{
public:
        virtual double value (const Point<dim>  &p,
                              const unsigned int component = 0) const;
};

/*///////////////////////////////////
 /////////// FUNCTIONS //////////////
 ////////////////////////////////////
 */

HeatEquation::HeatEquation (double x)
:
fe(1),
dof_handler(triangulation),
time_step(1. / 50),
time_end(x),
theta(0.5)
{}

void HeatEquation::setup_system()
{
        dof_handler.distribute_dofs(fe);
        
        std::cout << std::endl
        << "==========================================="
        << std::endl
        << "Number of active cells: " << triangulation.n_active_cells()
        << std::endl
        << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl
        << std::endl;
        
        CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, c_sparsity);
        sparsity_pattern.copy_from(c_sparsity);
        
        mass_matrix.reinit(sparsity_pattern);
        laplace_matrix.reinit(sparsity_pattern);
        system_matrix.reinit(sparsity_pattern);
        
        MatrixCreator::create_mass_matrix(dof_handler,
                                          QGauss<dim>(fe.degree+1),
                                          mass_matrix);/*,
                                          (const Function<dim> *)0,
                                          constraints);*/
        MatrixCreator::create_laplace_matrix(dof_handler,
                                             QGauss<dim>(fe.degree+1),
                                             laplace_matrix);/*,
                                             (const Function<dim> *)0,
                                             constraints);*/
        
        solution.reinit(dof_handler.n_dofs());
        old_solution.reinit(dof_handler.n_dofs());
        system_rhs.reinit(dof_handler.n_dofs());
        
        mass_matrix.print(cout);
        laplace_matrix.print(cout);
}

void HeatEquation::solve_time_step()
{
        SolverControl solver_control(1000, 1e-16 * system_rhs.l2_norm());
        SolverCG<> cg(solver_control);
        
        PreconditionSSOR<> preconditioner;
        preconditioner.initialize(system_matrix, 1.0);
        
        cg.solve(system_matrix, solution, system_rhs,
                 preconditioner);
        
        //constraints.distribute(solution);
        /*
         std::cout << "     " << solver_control.last_step()
         << " CG iterations." << std::endl;*/
}
/*
void HeatEquation::output_results() const
{
        DataOut<dim> data_out;
        
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "U");
        
        data_out.build_patches();
        
        const std::string filename = "output/solution-"
        + Utilities::int_to_string(timestep_number, 3) +
        ".vtk";
        std::ofstream output(filename.c_str());
        data_out.write_vtk(output);
}*/

void HeatEquation::build_matrix()
{
        Vector<double> tmp;
        Vector<double> forcing_terms;
        
        tmp.reinit (solution.size());
        forcing_terms.reinit (solution.size());
        
        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step, laplace_matrix);
        
        cout<<"before:\nmatrix @"<<time<<"\n";
        system_matrix.print(cout);
        cout<<"\n";
}

void HeatEquation::run()
{
        const unsigned int initial_global_refinement = 4;
        //const unsigned int n_adaptive_pre_refinement_steps = 4;
        
        GridGenerator::hyper_cube (triangulation, 0., 1.);
        triangulation.refine_global (initial_global_refinement);
        
        setup_system();
        /*
        std::ofstream out ("grid-heat.eps");
        GridOut grid_out;
        grid_out.write_eps (triangulation, out);
        */
        //unsigned int pre_refinement_step = 0;
        
        Vector<double> tmp;
        Vector<double> forcing_terms;
        
        //start_time_iteration:
        
        tmp.reinit (solution.size());
        forcing_terms.reinit (solution.size());
        
        
        VectorTools::interpolate(dof_handler,
                                 ZeroFunction<dim>(),
                                 old_solution);
        solution = old_solution;
        
        timestep_number = 0;
        time            = 0;
        
        output_results();
        
        build_matrix();
        
        while (time <= time_end)
        {
        
                time += time_step;
                ++timestep_number;
                
                std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;
                
                mass_matrix.vmult(system_rhs, old_solution); //system_rhs=mass*old_solution
                
                laplace_matrix.vmult(tmp, old_solution); //tmp=laplace*old_solution
                
                system_rhs.add(-(1 - theta) * time_step, tmp);
                
                // The second piece is to compute the contributions of the source
                // terms. This corresponds to the term $k_n
                // \left[ (1-\theta)F^{n-1} + \theta F^n \right]$. The following
                // code calls VectorTools::create_right_hand_side to compute the
                // vectors $F$, where we set the time of the right hand side
                // (source) function before we evaluate it. The result of this
                // all ends up in the forcing_terms variable:
                RightHandSide rhs_function;
                rhs_function.set_time(time);
                VectorTools::create_right_hand_side(dof_handler,
                                                    QGauss<dim>(fe.degree+1),
                                                    rhs_function,
                                                    tmp);
                forcing_terms = tmp;
                forcing_terms *= time_step * theta;
                
                rhs_function.set_time(time - time_step);
                VectorTools::create_right_hand_side(dof_handler,
                                                    QGauss<dim>(fe.degree+1),
                                                    rhs_function,
                                                    tmp);
                
                forcing_terms.add(time_step * (1 - theta), tmp);
                
                // Next, we add the forcing terms to the ones that
                // come from the time stepping, and also build the matrix
                // $M+k_n\theta A$ that we have to invert in each time step.
                // The final piece of these operations is to eliminate
                // hanging node constrained degrees of freedom from the
                // linear system:
                system_rhs += forcing_terms;
                
                //constraints.condense (system_matrix, system_rhs);
                
                // There is one more operation we need to do before we
                // can solve it: boundary values. To this end, we create
                // a boundary value object, set the proper time to the one
                // of the current time step, and evaluate it as we have
                // done many times before. The result is used to also
                // set the correct boundary values in the linear system:
                {
                        BoundaryValues boundary_values_function;
                        boundary_values_function.set_time(time);
                        
                        std::map<types::global_dof_index, double> boundary_values;
                        VectorTools::interpolate_boundary_values(dof_handler,
                                                                 0,
                                                                 boundary_values_function,
                                                                 boundary_values);
                        
                        MatrixTools::apply_boundary_values(boundary_values,
                                                           system_matrix,
                                                           solution,
                                                           system_rhs);
                }
                /*
                cout<<"after:\nmatrix @"<<time<<"\n";
                system_matrix.print(cout);
                cout<<"\n";
                cout<<"rhs @"<<time<<"\n";
                system_rhs.print(cout);
                cout<<"\n";
                */
                // With this out of the way, all we have to do is solve the
                // system, generate graphical data, and...
                solve_time_step();
                
                output_results();
                
                old_solution = solution;
        }
        cout<<"solution\n";
        solution.print(cout);
        cout<<"\n";
}

double RightHandSide::value (const Point<dim> &p,
                                  const unsigned int component) const
{
        return 1;
}

double BoundaryValues::value (const Point<dim> &/*p*/,
                              const unsigned int component) const
{
        Assert(component == 0, ExcInternalError());
        return 0;
}

/*////////////////////////////////////
 ////////////// MAIN /////////////////
 ////////////////////////////////////
 */

int main()
{
        HeatEquation x(1.);
        x.run();
	return 0;
}