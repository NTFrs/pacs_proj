#ifndef __integral_levy_price_hpp
#define __integral_levy_price_hpp

#include "LevyIntegralBase.hpp"
//! Class that handles the integral part with the Price transformation for a generic model
/*!
 * This class computes the integral parts of the equation when written with the \f$z=Se^y\f$ transformation in the integral part
 */
template<unsigned dim>
class LevyIntegralPrice: public LevyIntegralBase< dim > {
protected:
        unsigned order;
        unsigned order_max;
        double alpha_toll;
public:
        LevyIntegralPrice()=delete;
        
        LevyIntegralPrice(const LevyIntegralPrice &)=delete;
	//! Only constructor of the class
	/*!
         * Same as the constructor of the base class.
	 * \param lower_limit_ 		the left-bottom limit of the domain		
	 * \param upper_limit_ 		the rigth-upper limit of the domain
	 * \param Models_			A vector containing the needed models
         */
	LevyIntegralPrice(dealii::Point<dim> lower_limit_,
                          dealii::Point<dim> upper_limit_,
                          std::vector<Model *> & Models_,
                          unsigned order_)
        :
        LevyIntegralBase<dim>::LevyIntegralBase(lower_limit_, upper_limit_, Models_),
        order(order_),
        order_max(64),
        alpha_toll(constants::light_toll)
        {};
        
        //!
        /*! This function allows to set some adaptivity parameters
         * \param order_max_    Max number of integration nodes
         * \param alpha_toll_   Tollerance for alpha
         */
        virtual void set_adaptivity_params(unsigned order_max_,
                                           double alpha_toll_=constants::light_toll)
        {
                order_max=order_max_;
                alpha_toll=alpha_toll_;
        }
        
        LevyIntegralPrice& operator=(const LevyIntegralPrice &)=delete;
        
        //TODO add exception
        //! Computes the j part of the integrals
        /*!
         * This method computes the j part of the integrals and stores them inside j1 and j2 members. In a generic dimension does nothing,  it is only specialized for dimension 1 and 2.
         * \param sol			DealII Vector containing the values of the solutio function
         * \param dof_handler	DealII DoF Handler associated to this triangulation and solution
         * \param fe			DealII Finite elements associated to this triangulation and solution
         */
        virtual void compute_J(dealii::Vector<double> & sol,
                               dealii::DoFHandler<dim> & dof_handler,dealii::FE_Q<dim> & fe, std::vector< dealii::Point<dim> > const & vertices) {
                throw(std::logic_error("Compute_J not defined for this dimension\n"));
        }
};

//! Specialization of compute_J for 1 dimensional options
template<>
void LevyIntegralPrice<1>::compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<1> & dof_handler, dealii::FE_Q<1> & fe, std::vector< dealii::Point<1> > const & vertices) {
	
	using namespace dealii;
	//make j1 a vector of zeros with the same size as solution
	j1.reinit(dof_handler.n_dofs());
	
	//we need a quadrature formula on each cell,  as well as FEValues that handles FE and quadratures altogether
	QGauss<1> quadrature_formula(5);
	FEValues<1> fe_values(fe, quadrature_formula,  update_quadrature_points | update_values | update_JxW_values);
        
	const unsigned int n_q_points(quadrature_formula.size());
	
	//we declare an iterator over cells
	typename DoFHandler<1>::active_cell_iterator cell=dof_handler.begin_active(), endc=dof_handler.end();
        
        //this vector will contain the values of the solution in the present cell at quadrature points
	std::vector<double> sol_cell(n_q_points);
        
	//this will contain the quad points of the present cell
	std::vector< Point <1> > quad_points(n_q_points);
	
	double logz(0.);
        
	//we loop over cells. In each cell,  we get the values of the function as well as the quadrature points.
        for (;cell !=endc;++cell) {
		fe_values.reinit(cell);
		quad_points=fe_values.get_quadrature_points();
		fe_values.get_function_values(sol, sol_cell);
		//next we loop over all degrees of freedom and calculate the contribution of each cell to J[iter], where iter is the actual degree of freedom
		for (unsigned iter=0;iter<sol.size();++iter) {
                        Point<1> actual_vertex=vertices[iter];
                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                logz=log(quad_points[q_point](0)/actual_vertex(0));   j1[iter]+=fe_values.JxW(q_point)*sol_cell[q_point]*(*mods[0]).density(logz)/quad_points[q_point](0);
                        }
                        
                }
        }
        
        
        j_ran=true;
        
}

//! Specialization of compute_J for 2 dimensional options
template<>
void LevyIntegralPrice<2>::compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<2> & dof_handler, dealii::FE_Q<2> & fe, std::vector< dealii::Point<2> > const & vertices) {
	using namespace dealii;
	using namespace std;
        
        
        const unsigned N(dof_handler.n_dofs());
        
        //we clear these dealii Vectors to be all zero
	j1.reinit(N);
        j2.reinit(N);
        
        //we need a one dimensional quadrature formula, as well as the object that handles the values of FE on the faces of cells
	QGauss<1> quad1D(3);    
	FEFaceValues<2> fe_face(fe, quad1D, update_values  | update_quadrature_points | update_JxW_values);
        
	const unsigned int n_q_points=quad1D.size();
        
        //an iterator over cells and the final condition
	typename DoFHandler<2>::active_cell_iterator cell=dof_handler.begin_active(), endc=dof_handler.end();
        
        
	double z, karg;
        
	//vectors that will contain values of the solution and quadrature points on the current face 
	std::vector<Point <2> > quad_points(n_q_points);
	std::vector<double> sol_values(n_q_points);
        
	Point<2> actual_vertex;
	
        //we loop over all cells
        for (;cell !=endc;++cell) {
                
                //if this cell is at the left boundary,  we need to add the contribution to the vertices on this boundary.
		if (cell->face(0)->at_boundary()) {
                        
                        fe_face.reinit(cell,0);
                        quad_points=fe_face.get_quadrature_points();
                        
                        fe_face.get_function_values(sol,  sol_values);
                        
                        for (unsigned it=0;it<N;++it)
			{
                                actual_vertex=vertices[it];
                                if (fabs(actual_vertex(0)-this->lower_limit(0))<constants::grid_toll) {
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                z=quad_points[q_point](1);
                                                karg=log(z/actual_vertex(1));
                                                j2[it]+=fe_face.JxW(q_point)*((*mods[1]).density(karg))*(sol_values[q_point])/z;
                                        }
                                }
                                
			}
		}
		//same for the bottom boundary
		if (cell->face(2)->at_boundary()) {
                        
                        fe_face.reinit(cell, 2);
                        quad_points=fe_face.get_quadrature_points();
			fe_face.get_function_values(sol,  sol_values);
			
			for (unsigned it=0;it<N;++it)
                        {
				actual_vertex=vertices[it];
				if (fabs(actual_vertex(1)-this->lower_limit(1))<constants::grid_toll) {
					for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                z=quad_points[q_point](0);
                                                karg=log(z/actual_vertex(0));
                                                j1[it]+=fe_face.JxW(q_point)*((*mods[0]).density(karg))*(sol_values[q_point])/z;
					}
                                        
                                }
                                
                                
                        }
                }
		
		//now the upper side of the cell (number 3)
		{
                        //since this call is costly,  we save its value
                        double center(cell->face(3)->center()(1));
                        //we tell fe_face wich face we are on
                        fe_face.reinit(cell, 3);
                        
                        //we save the quadrature points and values of the solution since they will be used multiple times
                        quad_points=fe_face.get_quadrature_points();
                        fe_face.get_function_values(sol,  sol_values);
                        
                        //loop over all vertices
                        for (unsigned it=0;it<N;++it) {
                                actual_vertex=vertices[it];
                                //only the vertices whoose y coordiate is on this face get the contribute to j1[it]
                                if (fabs(actual_vertex(1)-center)<constants::grid_toll)
                                {
                                        //we finally calculate the contribute to j1
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                z=quad_points[q_point](0);
                                                karg=log(z/actual_vertex(0));
                                                j1[it]+=fe_face.JxW(q_point)*((*mods[0]).density(karg))*(sol_values[q_point])/z;
                                        }
                                }
                                
                        }
		}
		{	  
                        //the same but on the right face,  inverting x and y coordinates
                        double center(cell->face(1)->center()(0));
                        fe_face.reinit(cell, 1);
                        
                        quad_points=fe_face.get_quadrature_points();
                        
                        fe_face.get_function_values(sol,  sol_values);
                        
                        for (unsigned it=0;it<N;++it) {
                                actual_vertex=vertices[it];
                                if (fabs(actual_vertex(0)-center)<constants::grid_toll)
                                {
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                z=quad_points[q_point](1);
                                                karg=log(z/actual_vertex(1));
                                                j2[it]+=fe_face.JxW(q_point)*((*mods[1]).density(karg))*(sol_values[q_point])/z;
                                        }
                                }
                                
                        }
		}
                
                
		
        }
        
        j_ran=true;
        
}


# endif