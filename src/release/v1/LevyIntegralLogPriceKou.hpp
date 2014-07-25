#ifndef __levy_integral_log_price_kou__
# define __levy_integral_log_price_kou__

# include "LevyIntegralLogPrice.hpp"
# include "Quadrature.hpp"
# include <cmath>

template<unsigned dim>
class LevyIntegralLogPriceKou: public LevyIntegralLogPrice<dim> {
	
protected:
	std::vector<Quadrature_Laguerre> leftQuads;
	std::vector<Quadrature_Laguerre> rightQuads;
	bool adapting;
	
	virtual void setup_quadratures(unsigned n);
        virtual void compute_alpha();
        virtual void compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe);
	
public:
	LevyIntegralLogPriceKou()=delete;
	LevyIntegralLogPriceKou(dealii::Point<dim> lower_limit_,  dealii::Point<dim> upper_limit_,  std::vector<Model *> & Models_,  dealii::Function<dim> & BC_,  bool apt=true): LevyIntegralLogPrice<dim>::LevyIntegralLogPrice(lower_limit_, upper_limit_, Models_, BC_), adapting(apt) {
                if (!adapting)
                        this->setup_quadratures(16);
                else
                        this->setup_quadratures(2);
	}
};

template<unsigned dim>
void LevyIntegralLogPriceKou<dim>::setup_quadratures(unsigned int n)
{
	leftQuads.clear();rightQuads.clear();
        
	for (unsigned d=0;d<dim;++d) {
                leftQuads.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->get_lambda_m()));
                rightQuads.emplace_back(Quadrature_Laguerre(n, (this->Mods[d])->get_lambda_p()));
        }
        
}

template<unsigned dim>
void LevyIntegralLogPriceKou<dim>::compute_alpha() {
	this->alpha=std::vector<double>(dim, 0.);
        
        
        
	if (!adapting) {
                for (unsigned d=0;d<dim;++d) {
                        for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
                                this->alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
                                ((this->Mods[d])->get_p())*((this->Mods[d])->get_lambda())*
                                ((this->Mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
                        }
                        
                        for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
                                this->alpha[d]+=(exp(-(leftQuads[d].get_nodes())[i])-1)*
                                (1-((this->Mods[d])->get_p()))*((this->Mods[d])->get_lambda())*
                                ((this->Mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
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
                                for (unsigned i=0; i<rightQuads[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp((rightQuads[d].get_nodes())[i])-1)*
                                        ((this->Mods[d])->get_p())*((this->Mods[d])->get_lambda())*
                                        ((this->Mods[d])->get_lambda_p())*(rightQuads[d].get_weights())[i];
                                }
                                
                                for (unsigned i=0; i<leftQuads[d].get_order(); ++i) {
                                        this->alpha[d]+=(exp(-(leftQuads[d].get_nodes())[i])-1)*
                                        (1-((this->Mods[d])->get_p()))*((this->Mods[d])->get_lambda())*
                                        ((this->Mods[d])->get_lambda_m())*(leftQuads[d].get_weights())[i];
                                }
                        }
                        
                        setup_quadratures(2*leftQuads[0].get_order());
                        
                        
                        err=0.;
                        for (unsigned d=0;d<dim;++d)
                                err+=fabs(alpha_old[d]-(this->alpha[d]));
                        
                }
                while (err>constants::light_toll &&
                       rightQuads[0].get_order()<=order_max);
        }
        
}



template<unsigned dim>
void LevyIntegralLogPriceKou<dim>::compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe)
{/*
	using namespace dealii;
	unsigned N(sol.size());
	//TODO and if we do not initialise J nd do a pushback? 
	Vector<double> J;J.reinit(2*N);
	std::map<types::global_dof_index, Point<dim> > vertices;
	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);
        
	for (unsigned d=0;d<dim;++d) {
                tools::Solution_Trimmer<dim> func(d,this->boundary, dof_handler, sol, this->lower_limit, this->lower_limit);
                
                //#pragma omp parallel for
                for (unsigned int it=0;it<N;++it)
                {
                        std::vector< Point<dim> > quad_points(leftQuads[d].get_order()+rightQuads[d].get_order());
                        std::vector<double> f_u(leftQuads[d].get_order()+rightQuads[d].get_order());
                        
                        for (int i=0; i<quad_points.size(); ++i) {
                                quad_points[i][0]=this->quadrature_points_x[i][0] + this->grid_points[it][0];
                                quad_points[i][1]=this->grid_points[it][1];
                        }
                        //CONTROLLARE
                        for (; cell !=endc;++cell) {
                                
                                //reinit this 1D fevalues
                                fe_values2.reinit(cell);
                                //ATTENTION
                                //quadrature points are in 1D,  our functions take dimD
                                //Need to create a vector of 2D points
                                
                                //thus we get the 1D points
                                std::vector< Point<1> > quad_points_1D(fe_values2.get_quadrature_points());
                                
                                //and we create a vector to hold 2D points
                                std::vector< Point<dim> >
                                quad_points(n_q_points);
                                // This way,  the 1_i point of integration becomes (q_i, 0)
                                for (unsigned int q_point=0;q_point<n_q_points;++q_point) {
                                        quad_points[q_point][d]=quad_points_1D[q_point](0);
                                        quad_points[q_point](1-d)=0;}
                                std::vector<double> kern(n_q_points),  f_u(n_q_points);
                                
                                //and we compute the value of the density on that point (note the y coordinate is useless here) 
                                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                        kern[q_point]=(*this->Mods[d]).density(quad_points[q_point](d));
                                
                                //here we add the actual where we are, in order to obtain u(t, x_it+q_i, y_it)
                                //we have thus a vector of (q_i+x_it, y_it)
                                for (unsigned int q_point=0;q_point<n_q_points;++q_point)
                                        quad_points[q_point]+=vertices[it];
                                
                                //and we thus calculate the values of traslated u
                                func.value_list(quad_points, f_u);
                                
                                //and we can finally calculate the contribution to J_x(it)
                                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                        J(d*N+it)+=fe_values2.JxW(q_point)*kern[q_point]*f_u[q_point];
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
        }*/
}

#endif