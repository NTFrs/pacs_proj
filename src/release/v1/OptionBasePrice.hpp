#ifndef __option_base_price_hpp
#define __option_base_price_hpp

#include "OptionBase.hpp"

template <unsigned dim>
class OptionBasePrice: public OptionBase<dim> {
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
        OptionBasePrice(ExerciseType type_,
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
        OptionBasePrice(ExerciseType type_,
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
void OptionBasePrice<dim>::make_grid(){
        
        std::vector<unsigned> refinement(dim);
        
        for (unsigned i=0; i<dim; ++i) {
                
                this->Smin[i]=(1-this->f)*(*(this->models[i])).get_spot()*
                exp((this->r-(*(this->models[i])).get_vol()*(*(this->models[i])).get_vol()/2)*(this->T)
                    -(*(this->models[i])).get_vol()*sqrt(this->T)*6);
                
                this->Smax[i]=(1+this->f)*(*(this->models[i])).get_spot()*
                exp((this->r-(*(this->models[i])).get_vol()*(*(this->models[i])).get_vol()/2)*(this->T)
                    +(*(this->models[i])).get_vol()*sqrt(this->T)*6);
                
                refinement[i]=pow(2, this->refs-2);
        }
        
        dealii::GridGenerator::subdivided_hyper_rectangle (this->triangulation, refinement,
                                                   this->Smin, this->Smax);
        
        this->triangulation.refine_global(2);
        //TODO inutili vero? 
        this->grid_points=this->triangulation.get_vertices();
        
        return;
}

template <unsigned dim>
void OptionBasePrice<dim>::setup_integral(){
        
        if (this->model_type==OptionBase<dim>::ModelType::Kou) {
                std::cout<<"creo Kou\n";
                this->levy=std::unique_ptr<LevyIntegralBase<dim> > (new LevyIntegralPriceKou<dim>(this->Smin, this->Smax, this->models));
        }
        else if (this->model_type==OptionBase<dim>::ModelType::Merton) {
			this->levy=std::unique_ptr<LevyIntegralBase<dim> > (new LevyIntegralPriceMerton<dim>(this->Smin, this->Smax,this->models));
        }
}

template <unsigned dim>
void OptionBasePrice<dim>::assemble_system()
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
	dealii::FEValues<dim> fe_values (this->fe, quadrature_formula, update_values 
                                         | update_gradients | update_JxW_values | update_quadrature_points);
        
	const unsigned int   dofs_per_cell = (this->fe).dofs_per_cell;
	const unsigned int   n_q_points    = quadrature_formula.size();
        
	cout<< "Assembling System\n";
	cout<< "Degrees of freedom per cell: "<< dofs_per_cell<< endl;
	cout<< "Quadrature points per cell: "<< n_q_points<< endl;
        
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        
	dealii::FullMatrix<double> cell_ff(dofs_per_cell);
	dealii::FullMatrix<double> cell_mat(dofs_per_cell);
        
	typename DoFHandler<dim>::active_cell_iterator
	cell=(this->dof_handler).begin_active(),
	endc=(this->dof_handler).end();
	dealii::Tensor< 1 , dim, double > trasp;
	dealii::Tensor< 2 , dim, double > sig_mat;
	
	vector<dealii::Point<dim> > quad_points(n_q_points);
        
        for (; cell !=endc;++cell) {
                fe_values.reinit(cell);
                cell_ff=0;
                cell_mat=0;
                
                quad_points=fe_values.get_quadrature_points();
                
                for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                        
                        if (dim==1) {
                                trasp[0]=(this->r-(*(this->models[0])).get_vol()*(*(this->models[0])).get_vol()-alpha[0])*quad_points[q_point][0];
                                sig_mat[0][0]=0.5*(*(this->models[0])).get_vol()*(*(this->models[0])).get_vol()
                                *quad_points[q_point][0]*quad_points[q_point][0];
                        }
                        
                        else if (dim==2) {
                                
                                trasp[0]=-((*(this->models[0])).get_vol()*(*(this->models[0])).get_vol()
                                           *quad_points[q_point][0]
                                           +0.5*(this->rho)*(*(this->models[0])).get_vol()*(*(this->models[1])).get_vol()
                                           *quad_points[q_point][0]+(alpha[0]-this->r)*quad_points[q_point][0]);
                                
                                trasp[1]=-((*(this->models[1])).get_vol()*(*(this->models[1])).get_vol()*
                                           quad_points[q_point][1]+0.5*(this->rho)*(*(this->models[0])).get_vol()*
                                           (*(this->models[1])).get_vol()*quad_points[q_point][1]
                                           +(alpha[1]-this->r)*quad_points[q_point][1]);
                                
                                sig_mat[0][0]=0.5*(*(this->models[0])).get_vol()*(*(this->models[0])).get_vol()
                                *quad_points[q_point][0]*quad_points[q_point][0];
                                sig_mat[1][1]=0.5*(*(this->models[1])).get_vol()*(*(this->models[1])).get_vol()
                                *quad_points[q_point][1]*quad_points[q_point][1];
                                sig_mat[0][1]=0.5*(this->rho)*(*(this->models[0])).get_vol()*(*(this->models[1])).get_vol()*
                                quad_points[q_point][0]*quad_points[q_point][1];
                                sig_mat[1][0]=sig_mat[0][1];
                        }
                        
                        for (unsigned i=0;i<dofs_per_cell;++i)
                                for (unsigned j=0; j<dofs_per_cell;++j) {
                                        
                                        
                                        cell_mat(i, j)+=fe_values.JxW(q_point)*
                                        (
                                         (1/(this->dt)+(this->r)+lambda)*fe_values.shape_value(i, q_point)*fe_values.shape_value(j,q_point)
                                         +fe_values.shape_grad(i, q_point)*sig_mat*fe_values.shape_grad(j, q_point)
                                         -fe_values.shape_value(i, q_point)*trasp*fe_values.shape_grad(j, q_point)
                                         );
                                        
                                        cell_ff(i, j)+=fe_values.shape_value(i, q_point)*fe_values.shape_value(j, q_point)*fe_values.JxW(q_point);
                                        
                                }
                }
                
                cell->get_dof_indices (local_dof_indices);
                
                for (unsigned int i=0; i<dofs_per_cell;++i)
                        for (unsigned int j=0; j< dofs_per_cell; ++j) {
                                
                                ((this->system_matrix)).add(local_dof_indices[i], local_dof_indices[j], cell_mat(i, j));
                                (this->ff_matrix).add(local_dof_indices[i], local_dof_indices[j], cell_ff(i, j));
                                
                        }
        }
        
	(this->system_M2).add(1./(this->dt), this->ff_matrix);
        
        return;
        
}

template<unsigned dim>
double OptionBasePrice<dim>::get_price() {
        
	if (this->ran==false) {
                this->run();
        }
        
        dealii::Point<dim> p;
        
        for (unsigned i=0; i<dim; ++i) {
                p(i)=(*(this->models[i])).get_spot();
        }
        
	Functions::FEFieldFunction<dim> fe_function (this->dof_handler, this->solution);
	return fe_function.value(p);
}


#endif