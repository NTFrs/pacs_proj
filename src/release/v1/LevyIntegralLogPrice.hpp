#ifndef __levy_integral_log_price__
# define __levy_integral_log_price__

#include "LevyIntegralBase.hpp"

template<unsigned dim>
class LevyIntegralLogPrice: public LevyIntegralBase<dim> {
	
protected:
	//TODO if done like this can cause problems if Levy is called with LevyIntegralLogPrice(-.-.-.BC<>())
	std::unique_ptr<dealii::Function<dim> > boundary;
public:
        LevyIntegralLogPrice()=delete;
        
        LevyIntegralLogPrice(const LevyIntegralLogPrice &)=delete;
        
        LevyIntegralLogPrice& operator=(const LevyIntegralLogPrice &)=delete;
        
        LevyIntegralLogPrice(dealii::Point<dim> lower_limit_,
                             dealii::Point<dim> upper_limit_,
                             std::vector<Model *> & models_,
                             std::unique_ptr<dealii::Function<dim> > BC_)
        :
        LevyIntegralBase<dim>::LevyIntegralBase(lower_limit_, upper_limit_, models_),
        boundary(std::move(BC_))
        {};
	
	virtual void compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<dim> & dof_handler, dealii::FE_Q<dim> & fe);
	virtual inline void set_time(double tm) {boundary->set_time(tm);};
	
};

template<unsigned dim>
void LevyIntegralLogPrice<dim>::compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe)
{
	using namespace dealii;
	unsigned N(sol.size());
	//TODO and if we do not initialise J nd do a pushback? 
	Vector<double> J;J.reinit(2*N);
	std::map<types::global_dof_index, Point<dim> > vertices;
	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);
	
        for (unsigned d=0;d<dim;++d) {
                tools::Solution_Trimmer<dim> func(d,*this->boundary, dof_handler, sol, this->lower_limit, this->upper_limit);
		
                Triangulation<1> integral_triangulation;
                GridGenerator::subdivided_hyper_cube(integral_triangulation, pow(2, 5), this->bMin(d), this->bMax(d));
                
                FE_Q<1> fe_integral(1);
                DoFHandler<1> dof_integral(integral_triangulation);
                
                dof_integral.distribute_dofs(fe_integral);
                
                QGauss<1> quadrature_formula2(10);
                FEValues<1> fe_values2 (fe_integral, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);
                
                const unsigned int   n_q_points    = quadrature_formula2.size();
                //#pragma omp parallel for
                for (unsigned int it=0;it<N;++it)
                {
                        typename DoFHandler<1>::active_cell_iterator
                        cell=dof_integral.begin_active(),
                        endc=dof_integral.end();
                        
                        
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
                                }
                                std::vector<double> kern(n_q_points),  f_u(n_q_points);
                                
                                //and we compute the value of the density on that point (note the y coordinate is useless here) 
                                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                                        kern[q_point]=(*this->mods[d]).density(quad_points[q_point](d));
                                
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
        this->j1.reinit(N);
        for (unsigned i=0;i<this->j1.size();++i)
                this->j1[i]=J[i];
        if (dim==2) {
                this->j2.reinit(N);
                for (unsigned i=0;i<this->j2.size();++i)
                        this->j2[i]=J[i+N];
        }
        this->j_ran=true;
}


# endif