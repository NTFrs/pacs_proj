#ifndef __american_option_hpp
#define __american_option_hpp

#include "OptionBase.hpp"

template <unsigned dim>
class AmericanOption final: public OptionBase<dim>
{
private:
        ExerciseType us;
        
        virtual void solve();
public:
        // Constructor 1d
        AmericanOption(BlackScholesModel const &model,
                       double r_,
                       double T_,
                       double K_,
                       unsigned refs_,
                       unsigned time_step_)
        :
        OptionBase<dim>::OptionBase(ExerciseType::US, model, r_, T_, K_, refs_, time_step_),
        us(ExerciseType::US)
        {};
        
        // Cosntructor 2d
        AmericanOption(BlackScholesModel const &model1,
                       BlackScholesModel const &model2,
                       double r_,
                       double T_,
                       double K_,
                       unsigned refs_,
                       unsigned time_step_)
        :
        OptionBase<dim>::OptionBase(ExerciseType::US, model1, model2, r_, T_, K_, refs_, time_step_),
        us(ExerciseType::US)
        {};
        
        
};

template <unsigned dim>
void AmericanOption<dim>::solve ()
{
        VectorTools::interpolate (this->dof_handler,
                                  FinalCondition<dim>(this->K, OptionType::Put),
                                  this->solution);
        
        
        {
                DataOut<dim> data_out;
                
                data_out.attach_dof_handler (this->dof_handler);
                data_out.add_data_vector (this->solution, "begin");
                
                data_out.build_patches ();
                
                std::ofstream output ("begin.gpl");
                data_out.write_gnuplot (output);
        }
        
        cout<<"grid_points\n";
        for (unsigned i=0; i<this->grid_points.size(); ++i) {
                cout<<this->grid_points[i]<<"\t";
        }
        cout<<"\n";
        
	unsigned Step=this->time_step;
        
        BoundaryCondition<dim> bc(this->K, this->T,  this->r, OptionType::Put);
        
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
                
                unsigned maxiter=1000;
                double tollerance=constants::high_toll;
                
                Vector<double> solution_old=this->solution;
                
                bool converged=false;
                
                for (unsigned k=0; k<maxiter && !converged; ++k) {
                
                        (this->matrix_with_sor)->ProjectedSOR_step(this->solution,
                                                                   solution_old,
                                                                   this->system_rhs,
                                                                   this->grid_points,
                                                                   this->K);
                        
                        auto temp=this->solution;
                        temp.add(-1, solution_old);
                        
                        if (temp.linfty_norm()<tollerance){
                                converged=true;
                        }
                        
                        else
                                solution_old=this->solution;
                        
                        
                        if (k==maxiter-1) {
                                cout<<"Warning: maxiter reached.\n";
                        }
                        
                }
                
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