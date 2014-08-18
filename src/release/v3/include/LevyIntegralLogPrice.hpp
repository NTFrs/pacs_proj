#ifndef __levy_integral_logprice_hpp
#define __levy_integral_logprice_hpp

#include "LevyIntegralBase.hpp"

//! Class that handles the integral part with the LogPrice transformation for a generic model
/*!
 * This class computes the integral parts of the equation when written with the \f$x=log\left(\frac{S}{S_0}\right)\f$ transformation. It should be noted that, since this transformation requires a boundary condition,  it stores a unique_ptr to a Function object that is intended to be the boundary condition. The best way is thus to pass te boundary condition with the use of the move semantic.
 */
template<unsigned dim>
class LevyIntegralLogPrice: public LevyIntegralBase<dim> {
protected:
        std::unique_ptr<dealii::Function<dim> > boundary;
	bool adapting;
        bool adapted;
        unsigned order;
        unsigned order_max;
        double alpha_toll;
        double J_toll;
        
        virtual void setup_quadratures(unsigned n){
                order=n;
        };
        
	virtual double get_one_J(dealii::Point<dim> vert, tools::Solution_Trimmer<dim> & trim,  unsigned d);
public:
        LevyIntegralLogPrice()=delete;
        
        LevyIntegralLogPrice(const LevyIntegralLogPrice &)=delete;
        
        LevyIntegralLogPrice& operator=(const LevyIntegralLogPrice &)=delete;
        //! Only constructor of this class
        /*!
         * Similar to constructor of base class,  adds the space for a boundary condition.
         * \param lower_limit_ 		the left-bottom limit of the domain		
         * \param upper_limit_ 		the rigth-upper limit of the domain
         * \param Models_			A vector containing the needed models
         * \param BC_ 				Pointer to the Boundary Condition. Best to use std::move(BC),  where BC is std::unique_ptr to a dinamically allocated Function\<dim\> object from Deal.II (possibly a BoundaryConditionLogPrice)
         */
        LevyIntegralLogPrice(dealii::Point<dim> lower_limit_,
                             dealii::Point<dim> upper_limit_,
                             std::vector<Model *> & models_,
                             std::unique_ptr<dealii::Function<dim> > BC_,
                             bool apt=true)
        :
        LevyIntegralBase<dim>::LevyIntegralBase(lower_limit_, upper_limit_, models_), 
        boundary(std::move(BC_)),
        adapting(apt),
        adapted(false),
        order(4),
        order_max(64),
        alpha_toll(constants::light_toll),
        J_toll(constants::light_toll)
        {};
        
        //!
        /*! This function allows to set some adaptivity parameters
         * \param order_max_    Max number of integration nodes
         * \param alpha_toll_   Tollerance for alpha
         * \param J_toll_       Tollerance for J
         */
        virtual void set_adaptivity_params(unsigned order_max_,
                                           double alpha_toll_=constants::light_toll,
                                           double J_toll_=constants::light_toll)
        {
                order_max=order_max_;
                alpha_toll=alpha_toll_;
                J_toll=J_toll_;
        }
        
	//! Computes the J part of the integral for a logprice transformation
	/*!
	 * This method computes the j part of the integrals and stores them inside j1 and j2 members. It uses Gauss quadrature points,  so it works with any model.
	 * \param sol			DealII Vector containing the values of the solutio function
	 * \param dof_handler	DealII DoF Handler associated to this triangulation and solution
	 * \param fe			DealII Finite elements associated to this triangulation and solution
	 */
	virtual void compute_J(dealii::Vector<double> & sol,
                               dealii::DoFHandler<dim> & dof_handler,
                               dealii::FE_Q<dim> & fe, std::vector< dealii::Point<dim> > const & vertices)
        {
                throw(std::logic_error("Compute_J not defined in this dimension."));
        };
	
	//! Used to set the time for the boundary condition (if it depends on time)
	virtual inline void set_time(double tm) {boundary->set_time(tm);};
	
};

template<unsigned dim>
double LevyIntegralLogPrice<dim>::get_one_J(dealii::Point< dim > vert, tools::Solution_Trimmer< dim >& trim, unsigned d)
{
	using namespace dealii;
	double j(0);
	
	//we declare a one dimensional grid that will be used to calculate the integral
	Triangulation<1> integral_triangulation;
	GridGenerator::subdivided_hyper_cube(integral_triangulation, pow(2, 5), this->bMin(d), this->bMax(d));
        
	//and the dealii tools that handle the automatic scaling of quadrature to the present cell
	FE_Q<1> fe_integral(1);
	DoFHandler<1> dof_integral(integral_triangulation);
	dof_integral.distribute_dofs(fe_integral);
	QGauss<1> quadrature_formula2(order);
	FEValues<1> fe_values2 (fe_integral, quadrature_formula2, update_values | update_quadrature_points | update_JxW_values);
        
	const unsigned int   n_q_points    = quadrature_formula2.size();
        
	//we declare an iterator on the integral grid
	typename DoFHandler<1>::active_cell_iterator
	cell=dof_integral.begin_active(),
	endc=dof_integral.end();
        
	for (; cell !=endc;++cell) {
                
                //reinit this 1D fevalues
                fe_values2.reinit(cell);
                //ATTENTION
                //quadrature points are in 1D,  our functions take Point<dim>
                //Need to create a vector of dim-dimensional points
                
                //thus we extract the 1D points in the current cell
                std::vector< Point<1> > quad_points_1D(fe_values2.get_quadrature_points());
                
                //and we create a vector to hold dim-dimensional points
                std::vector< Point<dim> >
                quad_points(n_q_points);
                
                // This way,  the 1 dimensional point of quadrature is put in the d coordinate of the point
                for (unsigned int q_point=0;q_point<n_q_points;++q_point) {
                        quad_points[q_point][d]=quad_points_1D[q_point](0);
                }
                std::vector<double> kern(n_q_points),  f_u(n_q_points);
                
                //and we compute the value of the density on that point (note the coordinates different from d are useless here) 
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        kern[q_point]=(*this->mods[d]).density(quad_points[q_point](d));
                
                //here we add the actual point whe are computing, in order to obtain for example u(t, x_it+q_i, y_it)
                //we have thus a vector of (q_i+x_it, y_it)
                for (unsigned int q_point=0;q_point<n_q_points;++q_point)
                        quad_points[q_point]+=vert;
                
                //and we thus calculate the values of traslated u
                trim.value_list(quad_points, f_u);
                
                //and we can finally calculate the contribution to J_d(it)
                for (unsigned q_point=0;q_point<n_q_points;++q_point)
                        j+=fe_values2.JxW(q_point)*kern[q_point]*f_u[q_point];
                
        }
        return j;
}


template<>
void LevyIntegralLogPrice<1>::compute_J(dealii::Vector< double >& sol,
                                        dealii::DoFHandler<1>& dof_handler,
                                        dealii::FE_Q<1>& fe, std::vector< dealii::Point<1> > const & vertices)
{
	using namespace dealii;
	unsigned N(sol.size());
        
        j1.reinit(N);
        
        //the next class is used to return the value of the solution on the specified point,  if the point is inside the domain. Otherwise returns the boundary condition.
        tools::Solution_Trimmer<1> func(0,*this->boundary, dof_handler, sol, this->lower_limit, this->upper_limit);
        //thus,  for each node on the mesh we evaluate J(x_i)
        if (!adapting || adapted) {
#pragma omp parallel for
                for (unsigned int it=0;it<N;++it) {
                        this->j1(it)=this->get_one_J(vertices[it], func, 0);
                }
        }
        else {
                order=4;
                
                dealii::Vector<double> j1_old;
                double err;
                
                do  {
                        j1_old=this->j1;
                        j1.reinit(N);
                        
#pragma omp parallel for
                        for (unsigned int it=0;it<N;++it) {
                                this->j1(it)=this->get_one_J(vertices[it], func, 0);
                        }
                        
                        order=2*order;
                        setup_quadratures(order);
                        
                        auto temp=j1;
                        temp.add(-1, j1_old);
                        err=temp.linfty_norm();
                }
                while (err>J_toll && order<order_max);
                
        }
        
        this->j_ran=true;
}

template<>
void LevyIntegralLogPrice<2>::compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<2>& dof_handler, dealii::FE_Q<2>& fe, std::vector< dealii::Point<2> > const & vertices)
{
	using namespace dealii;
	unsigned N(sol.size());
        
	j1.reinit(N);
	j2.reinit(N);
        
        //the next class is used to return the value of the solution on the specified point,  if the point is inside the domain. Otherwise returns the boundary condition.
	tools::Solution_Trimmer<2> func1(0,*this->boundary, dof_handler, sol, this->lower_limit, this->upper_limit);
	tools::Solution_Trimmer<2> func2(1,*this->boundary, dof_handler, sol, this->lower_limit, this->upper_limit);
        //thus,  for each node on the mesh
        
        if (!adapting || adapted) {
# pragma omp parallel for
                for (unsigned int it=0;it<N;++it) {
                        this->j1(it)=this->get_one_J(vertices[it], func1, 0);
                        this->j2(it)=this->get_one_J(vertices[it], func2, 1);
                }
        }
        else {
                order=4;
                
                dealii::Vector<double> j1_old;
                dealii::Vector<double> j2_old;
                double err1;
                double err2;
                
                do  {
                        j1_old=this->j1;
                        j1.reinit(N);
                        
                        j2_old=this->j2;
                        j2.reinit(N);
                        
#pragma omp parallel for
                        for (unsigned int it=0;it<N;++it) {
                                this->j1(it)=this->get_one_J(vertices[it], func1, 0);
                                this->j2(it)=this->get_one_J(vertices[it], func2, 1);
                        }
                        
                        order=2*order;
                        setup_quadratures(order);
                        
                        auto temp1=j1;
                        temp1.add(-1, j1_old);
                        
                        auto temp2=j2;
                        temp2.add(-1, j2_old);
                        
                        err1=temp1.linfty_norm();
                        err2=temp2.linfty_norm();
                        
                }
                while ((err1>J_toll || err2>J_toll) && order<order_max);
                
                adapted=true;
                
        }
        
	this->j_ran=true;
}

# endif