#include "Levy.hpp"

template<unsigned dim>
class EuropeanCallUpOut: public OptionBasePrice<dim> {
protected:
        
        std::vector<double> h;
        
        virtual void make_grid();
        virtual void solve();
public:
        EuropeanCallUpOut()=delete;
        
        EuropeanCallUpOut(const EuropeanCallUpOut &)=delete;
        
        EuropeanCallUpOut(Model * const model,
                          double H_,
                          double r_,
                          double T_,
                          double K_,
                          unsigned refs_,
                          unsigned time_step_)
        :
        OptionBasePrice<dim>::OptionBasePrice(ExerciseType::EU, model, r_, T_, K_, refs_, time_step_)
        {
                h.push_back(H_);
        };
        
        EuropeanCallUpOut(Model * const model1,
                          Model * const model2,
                          double H1_,
                          double H2_,
                          double rho_,
                          double r_,
                          double T_,
                          double K_,
                          unsigned refs_,
                          unsigned time_step_)
        :
        OptionBasePrice<dim>::OptionBasePrice(ExerciseType::EU, model1, model2, rho_, r_, T_, K_, refs_, time_step_)
        {
                h.push_back(H1_);
                h.push_back(H2_);
        };
        
        EuropeanCallUpOut& operator=(const EuropeanCallUpOut &)=delete;
};

template<unsigned dim>
void EuropeanCallUpOut<dim>::make_grid(){
        
        using namespace dealii;
        
        std::vector<unsigned> refinement(dim);
        
        for (unsigned i=0; i<dim; ++i) {
                
                this->Smin[i]=(1-this->f)*(*(this->models[i])).get_spot()*
                exp((this->r-(*(this->models[i])).get_vol()*(*(this->models[i])).get_vol()/2)*(this->T)
                    -(*(this->models[i])).get_vol()*sqrt(this->T)*6);
                
                this->Smax[i]=h[i];
                
                refinement[i]=pow(2, this->refs-1);
        }
        
        dealii::GridGenerator::subdivided_hyper_rectangle (this->triangulation, refinement,
                                                           this->Smin, this->Smax);
        
        this->triangulation.refine_global();
        
        return;
}

template <unsigned dim>
void EuropeanCallUpOut<dim>::solve ()
{
        using namespace dealii;
        using namespace std;
        
        VectorTools::interpolate (this->dof_handler,
                                  FinalConditionPrice<dim>(this->K, OptionType::Call),
                                  this->solution);
        
        if (this->print) {
                this->print_solution_gnuplot("begin");
                this->print_solution_matlab("begin");
        }
        
	unsigned Step=this->time_step;
        
        if (dim==2 && this->print_grids) {
                this->print_grid(Step);
        }
        
	for (double time=this->T-this->dt; Step>0 ;time-=this->dt, --Step) {
                
                if (this->verbose) {
                        cout<< "Step "<<Step<<"\t at time "<<time<<"\n";
                }
                
                if (this->refine && Step%20==0 && Step!=this->time_step){
                        this->refine_grid();
                        if (dim==2 && this->print_grids) {
                                this->print_grid(Step);
                        }
                }
                //
                if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes) {
                        
                        Vector<double> *J_x;
                        Vector<double> *J_y;
                        Vector<double> temp;
                        
                        this->levy->compute_J(this->solution, this->dof_handler, this->fe, this->vertices);
                        
                        if (dim==1)
                                this->levy->get_j_1(J_x);
                        else
                                this->levy->get_j_both(J_x, J_y);
                        
                        (this->system_M2).vmult(this->system_rhs,this->solution);
                        
                        temp.reinit(this->dof_handler.n_dofs());
                        
                        (this->ff_matrix).vmult(temp, *J_x);
                        
                        this->system_rhs+=temp;
                        
                        if (dim==2) {
                                temp.reinit(this->dof_handler.n_dofs());
                                
                                this->ff_matrix.vmult(temp, *J_y);
                                
                                this->system_rhs+=temp;
                        }
                        
                }
                
                else
                        this->system_M2.vmult(this->system_rhs, this->solution);
                
                //
                
                {
                        
                        std::map<types::global_dof_index,double> boundary_values;
                        
                        VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                  0,
                                                                  ZeroFunction<dim>(),
                                                                  boundary_values);
                        
                        if (dim==1) {
                                VectorTools::interpolate_boundary_values (this->dof_handler,
                                                                          1,
                                                                          ZeroFunction<dim>(),
                                                                          boundary_values);
                        }
                        
                        MatrixTools::apply_boundary_values (boundary_values,
                                                            (this->system_matrix),
                                                            this->solution,
                                                            this->system_rhs, false);
                        
                }
                
                auto pointer=static_cast<SparseMatrix<double> *> (&(this->system_matrix));
                
                SparseDirectUMFPACK solver;
                solver.initialize(this->sparsity_pattern);
                solver.factorize(*pointer);
                solver.solve(this->system_rhs);
                
                this->solution=this->system_rhs;
                
                this->constraints.distribute(this->solution);
        }
        
        if (this->print) {
                this->print_solution_gnuplot("end");
                this->print_solution_matlab("end");
        }
        
}

int main() {
        /*
        {
                BlackScholesModel model(95., 0.120381);
                
                EuropeanCallUpOut<1> x(model.get_pointer(), 110., 0.0367, 1., 90., 12, 250);
                
                x.set_print(true);
                
                x.run();
                
                std::cout<<"The price of the option is "<<x.get_price()<<"\n";
        }
        
        {
                KouModel model(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
                
                EuropeanCallUpOut<1> x(model.get_pointer(), 110., 0.0367, 1., 90., 10, 100);
                
                x.set_print(true);
                
                x.run();
                
                std::cout<<"The price of the option is "<<x.get_price()<<"\n";
        }
        */
        {
                MertonModel model1(80., 0.2, 0.1, 0.4552, 0.258147);
                MertonModel model2(120., 0.1, -0.390078, 0.338796, 0.174814);
                
                EuropeanCallUpOut<2> x(model1.get_pointer(),
                                       model2.get_pointer(),
                                       110., 110.,
                                       -0.2, 0.1, 1., 200., 6, 10);
                
                x.set_print(true);
                x.set_timing(true);
                
                x.run();
                
                std::cout<<"finita la run\n";
                
                double prezz=x.get_price();
                
                std::cout<<"Ho preso il prezzo\n";
                
                std::cout<<"The price of the option is "<<x.get_price()<<"\n";
                
        }
        return 0;
}