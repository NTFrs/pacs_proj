#ifndef __levy_integral_logprice_merton__
# define __levy_integral_logprice_merton__

# include "LevyIntegralLogPrice.hpp"
# include "Quadrature.hpp"
# include <cmath>

template<unsigned dim>
class LevyIntegralLogPriceMerton: public LevyIntegralLogPrice<dim> {
        
protected:
	std::vector<Quadrature_Hermite> quadratures;
	bool adapting;
        
	virtual void setup_quadratures(unsigned n);
	virtual void compute_alpha();
	
public:
	LevyIntegralLogPriceMerton()=delete;
	LevyIntegralLogPriceMerton(dealii::Point<dim> lower_limit_,
                                   dealii::Point<dim> upper_limit_,
                                   std::vector<Model *> & Models_,
                                   std::unique_ptr<dealii::Function<dim> > BC_,
                                   bool apt=true)
        :
        LevyIntegralLogPrice<dim>::LevyIntegralLogPrice(lower_limit_, upper_limit_, Models_, std::move(BC_)), 
        adapting(apt) {
                if (!adapting)
                        this->setup_quadratures(16);
                else
                        this->setup_quadratures(2);
        }
        
	virtual void compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe);
        
};

template<unsigned dim>
void LevyIntegralLogPriceMerton<dim>::setup_quadratures(unsigned int n)
{
	quadratures.clear();
	using namespace std;
	for (unsigned d=0;d<dim;++d) {
                quadratures.emplace_back(Quadrature_Hermite(n, (this->Mods[d])->get_nu(), (this->Mods[d])->get_delta() ));
        }
}

template<unsigned dim>
void LevyIntegralLogPriceMerton<dim>::compute_alpha()
{
	this->alpha=std::vector<double>(dim, 0.);
        
	std::cerr<<"Im calculating the merton one\n";
        
	if (!adapting) {
                for (unsigned d=0;d<dim;++d) {
                        for (unsigned i=0; i<quadratures[d].get_order(); ++i) {
                                this->alpha[d]+=(exp((quadratures[d].get_nodes())[i])-1)*
                                ((this->Mods[d])->get_lambda())/(((this->Mods[d])->get_delta())*sqrt(2*constants::pi))
                                *(quadratures[d].get_weights())[i];
                        }
                }
        }
        
	else {
                unsigned order_max=64;
                
                std::vector<double> alpha_old;
                double err;
                do  {
                        alpha_old=this->alpha;
                        this->alpha=std::vector<double>(dim, 0.);
                        
                        for (unsigned d=0;d<dim;++d) {
                                for (unsigned i=0; i<quadratures[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp((quadratures[d].get_nodes())[i])-1)*
                                        ((this->Mods[d])->get_lambda())/(((this->Mods[d])->get_delta())*sqrt(2*constants::pi))
                                        *(quadratures[d].get_weights())[i];
                                }
                                
                        }
                        
                        setup_quadratures(2*quadratures[0].get_order());
                        
                        
                        err=0.;
                        for (unsigned d=0;d<dim;++d)
                                err+=fabs(alpha_old[d]-(this->alpha[d]));
                        
                }
                while (err>constants::light_toll &&
                       quadratures[0].get_order()<=order_max);
        }
        
        
}

template<unsigned dim>
void LevyIntegralLogPriceMerton<dim>::compute_J(dealii::Vector< double >& sol, dealii::DoFHandler< dim >& dof_handler, dealii::FE_Q< dim >& fe)
{
	using namespace dealii;
	unsigned N(sol.size());
	//TODO and if we do not initialise J nd do a pushback? 
	Vector<double> J;J.reinit(2*N);
	std::map<types::global_dof_index, Point<dim> > vertices;
	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);
        
	for (unsigned d=0;d<dim;++d) {
                tools::Solution_Trimmer<dim> func(d,*this->boundary, dof_handler, sol, this->lower_limit, this->lower_limit);
                
                //#pragma omp parallel for
                for (unsigned int it=0;it<N;++it)
                {
                        std::vector< Point<dim> > quad_points(quadratures[d].get_order());
                        std::vector<double> f_u(quadratures[d].get_order());
                        
                        for (unsigned i=0; i<quad_points.size(); ++i) {
                                quad_points[i][d]=this->quadratures[d].get_nodes()[i] + vertices[it][d];
                                if (dim==2) {
                                        quad_points[i][1-d]=vertices[it][1-d];
                                }
                        }
                        
                        
                        // valuto f_u in quad_points
                        func.value_list(quad_points, f_u);
                        
                        // Integro dividendo fra parte sinistra e parte destra dell'integrale
                        for (unsigned i=0;i<quadratures[d].get_order();++i) {
                                J[d*N+it]+=f_u[i]*((this->Mods[d])->get_lambda())/(((this->Mods[d])->get_delta())*sqrt(2*constants::pi))
                                *(quadratures[d].get_weights())[i];
                        }
                        
                        
                }
                
        }
	this->J1.reinit(N);
	for (unsigned i=0;i<this->J1.size();++i)
                this->J1[i]=J[i];
	if (dim==2) {
                this->J2.reinit(N);
                for (unsigned i=0;i<this->J1.size();++i)
                        this->J2[i]=J[i+N];
        }
        
        this->j_ran=true;

}


#endif