#ifndef __levy_integral_log_price__
# define __levy_integral_log_price__

#include "LevyIntegralBase.hpp"
//! Class that handles the integral part with the LogPrice transformation for a generic model
/*!
 * This class computes the integral parts of the equation when written with the \f$x=log\left(\frac{S}{S_0}\right)\f$ transformation. It should be noted that, since this transformation requires a boundary condition,  it stores a unique_ptr to a Function object that is intended to be the boundary condition. The best way is thus to pass te boundary condition with the use of the move semantic.
 */
template<unsigned dim>
class LevyIntegralLogPrice: public LevyIntegralBase<dim> {
	
protected:
	//TODO if done like this can cause problems if Levy is called with LevyIntegralLogPrice(-.-.-.BC<>())
	std::unique_ptr<dealii::Function<dim> > boundary;
	
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
                             std::unique_ptr<dealii::Function<dim> > BC_)
        :
        LevyIntegralBase<dim>::LevyIntegralBase(lower_limit_, upper_limit_, models_), boundary(std::move(BC_)) {};
        
	//! Computes the J part of the integral for a logprice transformation
	/*!
	 * This method computes the j part of the integrals and stores them inside j1 and j2 members. It uses Gauss quadrature points,  so it works with any model.
	 * \param sol			DealII Vector containing the values of the solutio function
	 * \param dof_handler	DealII DoF Handler associated to this triangulation and solution
	 * \param fe			DealII Finite elements associated to this triangulation and solution
	 */
	virtual void compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<dim> & dof_handler, dealii::FE_Q<dim> & fe);
	
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
	QGauss<1> quadrature_formula2(10);
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


template<unsigned dim>
void LevyIntegralLogPrice<dim>::compute_J(dealii::Vector< double >& sol, dealii::DoFHandler<dim>& dof_handler, dealii::FE_Q<dim>& fe)
{
	using namespace dealii;
	unsigned N(sol.size());

	Vector<double> J;J.reinit(2*N);
	//we initialize a mapping between the degrees of fredom index and the corresponding point
	std::map<types::global_dof_index, Point<dim> > vertices;
	DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof_handler, vertices);
		//then for each dimension we repeat the following
        for (unsigned d=0;d<dim;++d) {
				//the next class is used to return the value of the solution on the specified point,  if the point is inside the domain. Otherwise returns the boundary condition.
                tools::Solution_Trimmer<dim> func(d,*this->boundary, dof_handler, sol, this->lower_limit, this->upper_limit);
                //thus,  for each node on the mesh
                #pragma omp parallel for
                for (unsigned int it=0;it<N;++it)
                J(d*N+it)=this->get_one_J(vertices[it], func, d);
                
                
        }
        
        //we then transfer the computed values on j1 and j2
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