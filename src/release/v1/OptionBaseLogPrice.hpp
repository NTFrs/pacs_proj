#ifndef __option_base_logprice_hpp
#define __option_base_logprice_hpp

#include "OptionBase.hpp"
#include "BoundaryConditionsLogPrice.hpp"

template <unsigned dim>
class OptionBaseLogPrice: public OptionBase<dim> {
protected:
        virtual void make_grid();
        virtual void assemble_system();
        virtual void setup_integral();
        virtual void solve()=0;
public:
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
        
        virtual inline double get_price();
};

template<unsigned dim>
void OptionBaseLogPrice<dim>::make_grid(){
        
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
                
                refinement[i]=pow(2, this->refs);
        }
        
        GridGenerator::subdivided_hyper_rectangle (this->triangulation, refinement,
                                                   this->Smin, this->Smax);
        
        this->grid_points=this->triangulation.get_vertices();
        
        return;
}

template <unsigned dim>
void OptionBaseLogPrice<dim>::setup_integral(){
        //da capire cosa sono upper_limit, lower_limit
        
        /*
        if (this->model_type==OptionBase<dim>::ModelType::Kou) {
                std::cout<<"creo Kou\n";
                this->levy=new LevyIntegralLogPriceKou<dim>(this->models);
        }
        else if (this->model_type==OptionBase<dim>::ModelType::Merton) {
                this->levy=new LevyIntegralLogPriceMerton<dim>(this->models);
        }
        else
                this->levy=NULL;
         */
        this->levy=NULL;
}

template <unsigned dim>
void OptionBaseLogPrice<dim>::assemble_system()
{
        using namespace std;
        
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
        
	cout<< "Assembling System\n";
	cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
	cout<< "Quadrature points per cell: "<< n_q_points<< endl;
        
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
	Tensor< 1 , dim, double > trasp;
	Tensor< 2 , dim, double > sig_mat;
	
        std::vector<Point<dim> > quad_points(n_q_points);
        
        if (dim==1) {
                dealii::Tensor< 1 , dim, double > ones;
                for (unsigned i=0;i<dim;++i)
                        ones[i]=1;
                
                for (; cell !=endc;++cell) {
                        fe_values.reinit(cell);
                        cell_dd=0;
                        cell_fd=0;
                        cell_ff=0;
                        for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                for (unsigned i=0;i<dofs_per_cell;++i)
                                        for (unsigned j=0; j<dofs_per_cell;++j) {
                                                
                                                cell_dd(i, j)+=fe_values.shape_grad(i, q_point)*fe_values.shape_grad(j, q_point)*fe_values.JxW(q_point);
                                                cell_fd(i, j)+=fe_values.shape_value(i, q_point)*(ones*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);
                                                cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                                
                                        }
                        
                        cell->get_dof_indices (local_dof_indices);
                        
                        for (unsigned int i=0; i<dofs_per_cell;++i)
                                for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                        
                                        dd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_dd(i, j));
                                        fd_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_fd(i, j));
                                        (this->ff_matrix).add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                        
                                }
                        
                }
                
                double diff=(*(this->models[0])).get_vol()*(*(this->models[0])).get_vol()/2;
                double trasp=this->r-(*(this->models[0])).get_vol()*
                (*(this->models[0])).get_vol()/2-alpha[0];
                double reaz=-(this->r)-lambda;
                
                ((this->system_matrix)).add(1/(this->dt)-0.5*reaz, this->ff_matrix); 
                ((this->system_matrix)).add(0.5*diff, dd_matrix);
                ((this->system_matrix)).add(-0.5*trasp, fd_matrix);
                
                (this->system_M2).add(1/(this->dt)+0.5*reaz, this->ff_matrix); 
                (this->system_M2).add(-0.5*diff, dd_matrix);
                (this->system_M2).add(0.5*trasp, fd_matrix);
                
        }
        
        else {
                // building tensors
                Tensor< 2 , dim, double > sigma_matrix;
                
                sigma_matrix[0][0]=(*(this->models[0])).get_vol()*(*(this->models[0])).get_vol();
                sigma_matrix[1][1]=(*(this->models[1])).get_vol()*(*(this->models[1])).get_vol();
                sigma_matrix[0][1]=(*(this->models[0])).get_vol()*(*(this->models[1])).get_vol()*
                (this->rho);
                sigma_matrix[1][0]=(*(this->models[0])).get_vol()*(*(this->models[1])).get_vol()*
                (this->rho);
                /*
                 Tensor< 1 , dim, double > ones;
                 for (unsigned i=0;i<dim;++i)
                 ones[i]=1;
                 */
                Tensor< 1, dim, double > trasp;
                trasp[0]=this->r-(*(this->models[0])).get_vol()*(*(this->models[0])).get_vol()/2-alpha[0];
                trasp[1]=this->r-(*(this->models[1])).get_vol()*(*(this->models[1])).get_vol()/2-alpha[1];
                
                for (; cell !=endc;++cell) {
                        fe_values.reinit(cell);
                        cell_dd=0;
                        cell_fd=0;
                        cell_ff=0;
                        cell_system=0;
                        for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                for (unsigned i=0;i<dofs_per_cell;++i)
                                        for (unsigned j=0; j<dofs_per_cell;++j) {
                                                
                                                // mass matrix
                                                cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                                
                                                // system matrix
                                                cell_system(i, j)+=fe_values.JxW(q_point)*
                                                (0.5*fe_values.shape_grad(i, q_point)*sigma_matrix*fe_values.shape_grad(j, q_point)-
                                                 fe_values.shape_value(i, q_point)*(trasp*fe_values.shape_grad(j,q_point))+
                                                 (1/(this->dt)+this->r+lambda+lambda)*
                                                 fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point));
                                                
                                        }
                        
                        cell->get_dof_indices (local_dof_indices);
                        
                        for (unsigned int i=0; i<dofs_per_cell;++i)
                                for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                        
                                        (this->ff_matrix).add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                        ((this->system_matrix)).add(local_dof_indices[i], local_dof_indices[j], cell_system(i, j));
                                        
                                }
                        
                }
                
                (this->system_M2).add(1/(this->dt), this->ff_matrix);
        }
        
        return;
        
}

template<unsigned dim>
double OptionBaseLogPrice<dim>::get_price() {
        
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