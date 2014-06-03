#ifndef __european_option_hpp
#define __european_option_hpp

#include "OptionBase.hpp"

template <unsigned dim>
class EuropeanOption final: public OptionBase<dim>
{
private:
        OptionType type;
        ExerciseType eu;
        
        virtual void solve();
public:
        // Constructor 1d
        EuropeanOption(OptionType type_,
                       Model * const model,
                       double r_,
                       double T_,
                       double K_,
                       unsigned refs_,
                       unsigned time_step_)
        :
        OptionBase<dim>::OptionBase(ExerciseType::EU, model, r_, T_, K_, refs_, time_step_),
        type(type_),
        eu(ExerciseType::EU)
        {};
        
        // Cosntructor 2d
        EuropeanOption(OptionType type_,
                       Model * const model1,
                       Model * const model2,
                       double rho_,
                       double r_,
                       double T_,
                       double K_,
                       unsigned refs_,
                       unsigned time_step_)
        :
        OptionBase<dim>::OptionBase(ExerciseType::EU, model1, model2, rho_, r_, T_, K_, refs_, time_step_),
        type(type_),
        eu(ExerciseType::EU)
        {};
        
        
};

template <unsigned dim>
void EuropeanOption<dim>::solve ()
{
        VectorTools::interpolate (this->dof_handler,
                                  FinalCondition<dim>(this->K, this->type),
                                  this->solution);
                
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (this->dof_handler);
                data_out.add_data_vector (this->solution, "begin");
                
                data_out.build_patches ();
                
                std::ofstream output ("begin.gpl");
                data_out.write_gnuplot (output);
        }
        
	unsigned Step=this->time_step;
        
        BoundaryCondition<dim> bc(this->K, this->T,  this->r, this->type);
        
	cout<< "time step is"<< this->time_step << endl;
	
	for (double time=this->T-this->dt;time >=0;time-=this->dt, --Step) {
                
                cout<< "Step "<< Step<<"\t at time \t"<< time<< endl;
                this->system_M2.vmult(this->system_rhs, this->solution);
                
                bc.set_time(time);
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        
                        VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                  0,
                                                                  bc,
                                                                  boundary_values);

                        MatrixTools::apply_boundary_values (boundary_values,
                                                            *(this->system_matrix),
                                                            this->solution,
                                                            this->system_rhs, false);
                        
                }
                
                SparseDirectUMFPACK solver;
                solver.initialize(this->sparsity_pattern);
                solver.factorize(*(this->system_matrix));
                solver.solve(this->system_rhs);
                
                this->solution=this->system_rhs;
                
        }
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (this->dof_handler);
                data_out.add_data_vector (this->solution, "end");
                
                data_out.build_patches ();
                
                std::ofstream output ("end.gpl");
                data_out.write_gnuplot (output);
        }
        
	this->ran=true;
        
}

#endif