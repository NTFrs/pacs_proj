#ifndef __option_base_logprice_hpp
#define __option_base_logprice_hpp

#include "OptionBase.hpp"

//! Class to handle option-type objects, with LogPrice transformation
/*! This class implements some methods inherited by LogPrice's option-type objects: make_grid, assemble_system and some output functions.
 */
template <unsigned dim>
class OptionBaseLogPrice: public OptionBase<dim> {
protected:
        virtual void make_grid();
        virtual void assemble_system();
        virtual void solve()=0;
        virtual void print_solution_matlab(std::string name_);
public:
        OptionBaseLogPrice()=delete;
        
        OptionBaseLogPrice(const OptionBaseLogPrice &)=delete;
        
        //! Constructor 1d
        /*!
         * Constructor 1d called by inherited classes.
         */
        OptionBaseLogPrice(ExerciseType type_,
                           Model * const model,
                           double r_,
                           double T_,
                           double K_,
                           unsigned refs_,
                           unsigned time_step_):
        OptionBase<dim>::OptionBase(type_, model, r_, T_, K_, refs_, time_step_)
        {};
        
        //! Cosntructor 2d
        /*!
         * Constructor 2d called by inherited classes.
         */
        OptionBaseLogPrice(ExerciseType type_,
                           Model * const model1,
                           Model * const model2,
                           double rho_,
                           double r_,
                           double T_,
                           double K_,
                           unsigned refs_,
                           unsigned time_step_)
        :
        OptionBase<dim>::OptionBase(type_, model1, model2, rho_, r_, T_, K_, refs_, time_step_)
        {};
        
        OptionBaseLogPrice& operator=(const OptionBaseLogPrice &)=delete;
        
        virtual double get_price();
};

template<unsigned dim>
void OptionBaseLogPrice<dim>::make_grid(){
        
        using namespace dealii;
        
        std::vector<unsigned> refinement(dim);
        
        for (unsigned i=0; i<dim; ++i) {
                
                this->Smin[i]=log((1-this->f)*(*(this->models[i])).get_spot()*
                                  exp((this->r-(*(this->models[i])).get_vol()*(*(this->models[i])).get_vol()/2)*
                                      (this->T)-(*(this->models[i])).get_vol()*sqrt(this->T)*6)/
                                  ((*(this->models[i])).get_spot()));
                
                this->Smax[i]=log((1+this->f)*(*(this->models[i])).get_spot()*
                                  exp((this->r-(*(this->models[i])).get_vol()*(*(this->models[i])).get_vol()/2)*
                                      (this->T)+(*(this->models[i])).get_vol()*sqrt(this->T)*6)/
                                  ((*(this->models[i])).get_spot()));
                
                refinement[i]=pow(2, this->refs-1);
        }
        
        GridGenerator::subdivided_hyper_rectangle (this->triangulation, refinement,
                                                   this->Smin, this->Smax);
        this->triangulation.refine_global();
        
        return;
}

template <unsigned dim>
void OptionBaseLogPrice<dim>::assemble_system()
{
        using namespace std;
        using namespace dealii;
        
        std::vector<double> alpha(dim,0.);
        
        if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes)
                this->levy->get_alpha(alpha);
        
        double lambda=0.;
        if (this->model_type!=OptionBase<dim>::ModelType::BlackScholes)
                for (unsigned i=0; i<dim; ++i)
                        lambda+=this->models[i]->get_lambda();
        
        dealii::QGauss<dim> quadrature_formula(2*dim);
        dealii::FEValues<dim> fe_values (this->fe, quadrature_formula, update_values |
                                         update_gradients |
                                         update_JxW_values | update_quadrature_points);
        
	const unsigned int   dofs_per_cell = (this->fe).dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
        
        if (this->verbose) {
                cout<< "Assembling System...\n";
                cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
                cout<< "Quadrature points per cell: "<< n_q_points<< endl;
        }
        
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        dealii::FullMatrix<double> cell_dd(dofs_per_cell);
	dealii::FullMatrix<double> cell_fd(dofs_per_cell);
	dealii::FullMatrix<double> cell_ff(dofs_per_cell);
	dealii::FullMatrix<double> cell_system(dofs_per_cell);
        
        dealii::SparseMatrix<double>            dd_matrix;
	dealii::SparseMatrix<double>            fd_matrix;
        
        dd_matrix.reinit(this->sparsity_pattern);
	fd_matrix.reinit(this->sparsity_pattern);
        
	typename dealii::DoFHandler<dim>::active_cell_iterator
	cell=(this->dof_handler).begin_active(),
	endc=(this->dof_handler).end();
	
        std::vector<Point<dim> > quad_points(n_q_points);
        
        dealii::Tensor<1, dim, double> trasp;
        Tensor<2, dim, double> sigma_matrix;
        
        Point<dim> dummy_point;
        
        for (unsigned d=0;d<dim;++d) {
                trasp[d]=this->r-(*(this->models[d])).get_vol()*(*(this->models[d])).get_vol()/2-alpha[d];
                dummy_point(d)=1.;
        }
        
        tools::make_diff<dim>(sigma_matrix, this->models, this->rho, dummy_point);
        
        for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                cell_ff=0;
                cell_system=0;
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        // mass matrix
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                        // system matrix
                                        cell_system(i, j)+=fe_values.JxW(q_point)*
                                        (fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)-
                                         fe_values.shape_value(i, q_point)*(trasp*fe_values.shape_grad(j,q_point))+
                                         (1/(this->dt)+this->r+lambda)*
                                         fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point));
                                        
                                }
		
                cell->get_dof_indices (local_dof_indices);
                auto pointer=static_cast<SparseMatrix<double> *> (&(this->system_matrix));
                this->constraints.distribute_local_to_global(cell_system, local_dof_indices, *pointer);
                this->constraints.distribute_local_to_global(cell_ff, local_dof_indices, this->ff_matrix);
                /*
                 for (unsigned int i=0; i<dofs_per_cell;++i)
                 for (unsigned int j=0; j< dofs_per_cell; ++j) {
                 
                 (this->ff_matrix).add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                 ((this->system_matrix)).add(local_dof_indices[i], local_dof_indices[j], cell_system(i, j));
                 
                 }
                 */
                
        }
	(this->system_M2).add(1/(this->dt), this->ff_matrix);
        
        if (this->verbose) {
                cout<<"Done!\n";
        }
        
        return;
        
}

template<unsigned dim>
void OptionBaseLogPrice<dim>::print_solution_matlab(std::string name_) {
        
        std::string name("matlab/");
        name.append(name_);
        name.append(std::to_string(this->id));
        name.append(".m");
        
        std::ofstream stream;
        stream.open(name);
        
        if (stream.is_open()) {
                if (dim==1) {
                        stream<<"grid=[ ";
                        for (unsigned i=0; i<this->solution.size()-1; ++i) {
                                stream<<this->models[0]->get_spot()*exp(this->vertices[i][0])<<"; ";
                        }
                        stream<<this->models[0]->get_spot()*
                        exp(this->vertices[this->solution.size()-1][0])<<" ];\n";
                }
                else {
                        stream<<"grid_x=[ ";
                        for (unsigned i=0; i<this->solution.size()-1; ++i) {
                                stream<<this->models[0]->get_spot()*exp(this->vertices[i][0])<<"; ";
                        }
                        stream<<this->models[0]->get_spot()*
                        exp(this->vertices[this->solution.size()-1][0])<<" ];\n";
                        stream<<"grid_y=[ ";
                        for (unsigned i=0; i<this->solution.size()-1; ++i) {
                                stream<<this->models[1]->get_spot()*exp(this->vertices[i][1])<<"; ";
                        }
                        stream<<this->models[1]->get_spot()*exp(this->vertices[this->solution.size()-1][1])<<" ];\n";
                }
                
                stream<<"sol=[ ";
                for (unsigned i=0; i<this->solution.size()-1; ++i) {
                        stream<<this->solution(i)<<"; ";
                }
                stream<<this->solution(this->solution.size()-1)<<" ];\n";
        }
        else {
                throw(std::ios_base::failure("Unable to open the file."));
        }
        
        stream.close();
        
}

template<unsigned dim>
double OptionBaseLogPrice<dim>::get_price() {
        
        using namespace dealii;
        
	if (this->ran==false) {
                this->run();
        }
        
        dealii::Point<dim> p;
        
        for (unsigned i=0; i<dim; ++i) {
                p(i)=0.;
        }
        
	dealii::Functions::FEFieldFunction<dim> fe_function (this->dof_handler, this->solution);
	return fe_function.value(p);
}

#endif