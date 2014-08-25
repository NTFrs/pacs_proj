#include <iostream>
#include "Factory.hpp"

template<unsigned dim>
class EuropeanLogPrice_Doubling: public EuropeanOptionLogPrice<dim> {
	
	protected:
	virtual void refine_grid();
	virtual void solve();
	void solve_one_step(dealii::Function<dim> & bc);
	void estimate_doubling(dealii::Vector< float >& errors);
	
	//need this since refine grid has no time input,  but estimate doubling needs it
	double actual_time;
	
	public:
	 EuropeanLogPrice_Doubling()=delete;
	 EuropeanLogPrice_Doubling(const EuropeanLogPrice_Doubling &)=delete;

	EuropeanLogPrice_Doubling(OptionType type_, Model * const model, double r_, double T_,	double K_,	unsigned refs_,	unsigned time_step_) :EuropeanOptionLogPrice<dim>::EuropeanOptionLogPrice(type_, model, r_, T_, K_, refs_, time_step_) {};
	
	EuropeanLogPrice_Doubling(OptionType type_, Model * const model1, Model * const model2, double rho_, double r_, double T_,	double K_,	unsigned refs_,	unsigned time_step_) :EuropeanOptionLogPrice<dim>::EuropeanOptionLogPrice(type_, model1, model2, rho_, r_, T_, K_, refs_, time_step_),  actual_time(1.) {};
	
	
};


//pretty much the same as solve of european,  except we strip its core and put it in solve_one_step. Also we updae actua_time
template<unsigned dim>
void EuropeanLogPrice_Doubling<dim>::solve() {
	using namespace dealii;
	using namespace std;

	std::vector<double> S0(dim);
	for (unsigned d=0; d<dim; ++d) {
	 S0[d]=this->models[d]->get_spot();
   }

	VectorTools::interpolate (this->dof_handler,
	 FinalConditionLogPrice<dim>(S0, this->K, this->type2),
	 this->solution);

	if (this->print) {
	 this->print_solution_gnuplot("begin");
	 this->print_solution_matlab("begin");
   }

	unsigned Step=this->time_step;

	BoundaryConditionLogPrice<dim> bc(S0, this->K, this->T,  this->r, this->type2);

	if (dim==2 && this->print_grids) {
	 this->print_grid(Step);
   }

	for (actual_time=this->T-this->dt; Step>0 ;actual_time-=this->dt, --Step) {

	 if (this->verbose) {
	  cout<< "Step "<<Step<<"\t at time "<<actual_time<<"\n";
	}

	 if (this->refine && Step%20==0 && Step!=this->time_step) {
	  this->refine_grid();
	  if (dim==2 && this->print_grids) {
	   this->print_grid(Step);
	 }
	}
	 //we put the rest in a solve_one_step
	 solve_one_step(bc);
	 
   }

	if (this->print) {
	 this->print_solution_gnuplot("end");
	 this->print_solution_matlab("end");
   }

  }

//core of solve goes here now,  needs the bc and actual_time (which is a member of the class anyway)
template<unsigned dim>
void EuropeanLogPrice_Doubling<dim>::solve_one_step(dealii::Function<dim> & bc) {
	using namespace dealii;

	if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes) {

	
	 Vector<double> *J_x;
	 Vector<double> *J_y;
	 Vector<double> temp;
	 this->levy->set_time(actual_time);
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

	bc.set_time(actual_time);

	{

	 std::map<types::global_dof_index,double> boundary_values;

	 VectorTools::interpolate_boundary_values (this->dof_handler,
	  0,
	  bc,
	  boundary_values);

	 if (dim==1) {
	  VectorTools::interpolate_boundary_values (this->dof_handler,
	   1,
	   bc,
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

//this is the real addon
//Basically we save our old option status (triangulatio, dof_handler, fe and solution) and compute a new solution doubling the mesh. We project it on the old mesh and compute the error in each cell.
template<unsigned dim>
void EuropeanLogPrice_Doubling<dim>::estimate_doubling(dealii::Vector< float >& errors)
{
	//same namespaces as always
	using namespace dealii;
	using namespace std;

	
	//we define some copies to the triangulation and all the dealii stuff
	Triangulation<dim> old_tria;
	//there is no copyconstructor to trangulation,  so we use this
	old_tria.copy_triangulation(this->triangulation);
	FE_Q<dim> old_fe(1);
	DoFHandler<dim> old_dof(old_tria);
	old_dof.distribute_dofs(old_fe);
	Vector<double> old_solution=this->solution;
	
	{
	 //we use a field describing our solution to interpolate a solution to the more refined mesh 
	 Functions::FEFieldFunction<dim>	moveSol(old_dof,  old_solution);

	 this->triangulation.refine_global(1);
	 this->setup_system();
	 VectorTools::interpolate(this->dof_handler, moveSol, this->solution);
	}
	//and we set up a new system
	this->assemble_system();
	//we solve here only a time step without moving the time
	{
	 //yet to do it we need a BC
	std::vector<double> S0(dim);
	for (unsigned d=0; d<dim; ++d) {
	 S0[d]=this->models[d]->get_spot();
    }
	BoundaryConditionLogPrice<dim> bc(S0, this->K, this->T,  this->r, this->type2);
	//here we solve
	solve_one_step(bc);
	}
	//now we have a newly computed solution on the refined mesh,  we want to interpolate it to the old mesh and we do it in the same way as before
	{
	 Functions::FEFieldFunction<dim> moveSol(this->dof_handler, this->solution); 

	 Vector<double> temp(old_dof.n_dofs());
	 VectorTools::interpolate(old_dof, moveSol, temp);
	 //now solution is the interpolation of the solution computed on a refined mesh
	 this->solution=temp;
	}
	//and we re-establish the status of our old option (except for solution,  we still need that)
	this->triangulation.clear();
	this->triangulation.copy_triangulation(old_tria);
	this->setup_system();
	this->assemble_system();

	//we can finally calculate the quadratic error on each cell
	//for that we need a cell iterator
	typename DoFHandler<dim>::active_cell_iterator cell=this->dof_handler.begin_active(),  endc=this->dof_handler.end();
	vector<types::global_dof_index> local_dof_indices(this->fe.dofs_per_cell);
	//the vector of errors
	errors.reinit(old_tria.n_active_cells());
	double err(0);
	unsigned ind(0),  count(0);
	
	for (;cell !=endc;++cell) {
	 err=0;
	 cell->get_dof_indices(local_dof_indices);
	 //on each cell,  we compute the error of every vertex of the cell
	 for (unsigned i=0;i<this->fe.dofs_per_cell;++i) {
	  ind=local_dof_indices[i];
	  err+=(this->solution[ind]-old_solution[ind])*(this->solution[ind]-old_solution[ind]);
	}
	 errors[count]=(err);
	 count++;
   }
	//and we finally re establish the satus of the solution
	this->solution=old_solution;
}

//just substituted kelly with estimate_doubling,  rest is equal to the one in OptionBase
template<unsigned dim>
void EuropeanLogPrice_Doubling<dim>::refine_grid() {
	using namespace dealii;

	Vector<float> estimated_error_per_cell (this->triangulation.n_active_cells());
	estimate_doubling(estimated_error_per_cell);

	GridRefinement::refine_and_coarsen_fixed_number (this->triangulation,
	 estimated_error_per_cell,
	 this->refine_index, this->coarse_index);

	SolutionTransfer<dim> solution_trans(this->dof_handler);
	Vector<double> previous_solution;
	previous_solution = this->solution;
	this->triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_solution);

	this->triangulation.execute_coarsening_and_refinement();
	this->setup_system ();

	solution_trans.interpolate(previous_solution, this->solution);
	this->assemble_system();
}



int main() {
	

	using namespace dealii;
	using namespace std;
	
// 	BlackScholesModel model(95., 0.120381);
	KouModel model(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
	
	MertonModel model1(80., 0.2, 0.1, 0.4552, 0.258147);
	MertonModel model2(120., 0.1, -0.390078, 0.338796, 0.174814);
	
	//this option is not on the factory,  so we declare it
	EuropeanLogPrice_Doubling<1> optie(OptionType::Call, model.get_pointer(), 0.0367, 1., 90., 10, 100);
	optie.set_refine_status(true, 0.03, 0.1);
	optie.set_integral_adaptivity_params(false,16);
	optie.run();
	
	cout<< "And the price is "<< optie.get_price()<< endl;
	cout<< "my target price is 12.427 \n";

	return 0;
}
