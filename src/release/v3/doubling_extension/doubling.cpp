#include <iostream>
#include "Factory.hpp"

template<unsigned dim>
class EuropeanLogPrice_Doubling: public EuropeanOptionLogPrice<dim> {
	
	protected:
	
	void solve_one_step(double time);
	void estimate_doubling(double time, dealii::Vector< float >& errors);
	virtual void refine_grid();
	
	public:
	 EuropeanLogPrice_Doubling()=delete;
	 EuropeanLogPrice_Doubling(const EuropeanLogPrice_Doubling &)=delete;

	EuropeanLogPrice_Doubling(OptionType type_, Model * const model, double r_, double T_,	double K_,	unsigned refs_,	unsigned time_step_) :EuropeanOptionLogPrice<dim>::EuropeanOptionLogPrice(type_, model, r_, T_, K_, refs_, time_step_) {};
	
	EuropeanLogPrice_Doubling(OptionType type_, Model * const model1, Model * const model2, double rho_, double r_, double T_,	double K_,	unsigned refs_,	unsigned time_step_) :EuropeanOptionLogPrice<dim>::EuropeanOptionLogPrice(type_, model1, model2, rho_, r_, T_, K_, refs_, time_step_) {};
	
	
};

template<unsigned dim>
void EuropeanLogPrice_Doubling<dim>::solve_one_step(double time) {
	
}

template<unsigned dim>
void EuropeanLogPrice_Doubling<dim>::estimate_doubling(double time, dealii::Vector< float >& errors)
{
	using namespace dealii;
	using namespace std;

	Triangulation<dim> old_tria;
	old_tria.copy_triangulation(triangulation);
	FE_Q<dim> old_fe(1);
	DoFHandler<dim> old_dof(old_tria);
	old_dof.distribute_dofs(old_fe);
	Vector<double> old_solution=this->solution;
	{
	 Functions::FEFieldFunction<dim>	moveSol(old_dof,  old_solution);

	 this->triangulation.refine_global(1);
	 this->setup_system();
	 VectorTools::interpolate(this->dof_handler, moveSol, this->solution);
	}
	this->assemble_system();
	//TODO need a solve_one_step here if using this
	solve_one_step(time);
	{
	 Functions::FEFieldFunction<dim> moveSol(dof_handler, solution); 

	 Vector<double> temp(old_dof.n_dofs());
	 VectorTools::interpolate(old_dof, moveSol, temp);
	 this->solution=temp;
	}
	this->triangulation.clear();
	this->triangulation.copy_triangulation(old_tria);
	this->setup_system();
	this->assemble_system();

	typename DoFHandler<dim>::active_cell_iterator cell=this->dof_handler.begin_active(),  endc=this->dof_handler.end();
	vector<types::global_dof_index> local_dof_indices(this->fe.dofs_per_cell);
	errors.reinit(old_tria.n_active_cells());
	double err(0);
	unsigned ind(0),  count(0);
	for (;cell !=endc;++cell) {
	 err=0;
	 cell->get_dof_indices(local_dof_indices);
	 for (unsigned i=0;i<fe.dofs_per_cell;++i) {
	  ind=local_dof_indices[i];
	  err+=(this->solution[ind]-old_solution[ind])*(this->solution[ind]-old_solution[ind]);
	}
	 errors[count]=(err);
	 count++;
   }

	this->solution=old_solution;
}