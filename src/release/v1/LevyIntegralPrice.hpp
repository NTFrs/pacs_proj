#ifndef __integral_levy_price_hpp
# define __integral_levy_price_hpp

#include "LevyIntegralBase.hpp"

template<unsigned dim>
class LevyIntegralPrice: public LevyIntegralBase< dim > {
public:
        LevyIntegralPrice()=delete;
        
        LevyIntegralPrice(const LevyIntegralPrice &)=delete;
	
	LevyIntegralPrice(dealii::Point<dim> lower_limit_,
                          dealii::Point<dim> upper_limit_,
                          std::vector<Model *> & Models_)
        :
        LevyIntegralBase<dim>::LevyIntegralBase(lower_limit_, upper_limit_, Models_)
        {};
        
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
                               dealii::DoFHandler<dim> & dof_handler,
                               dealii::FE_Q<dim> & fe) {
                std::cerr<<"Not defined for this dimension"<< std::endl;
        }
};

//! Specialization of compute_J for 1 dimensional options
template<>
void LevyIntegralPrice<1>::compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<1> & dof_handler, dealii::FE_Q<1> & fe) {
	
	using namespace dealii;
	//make j1 a vector of zeros with the same size as solution
	j1.reinit(sol.size());
	
	QGauss<1> quadrature_formula(5);
	FEValues<1> fe_values(fe, quadrature_formula,  update_quadrature_points | update_values | update_JxW_values);
        
	const unsigned int n_q_points(quadrature_formula.size());
	typename DoFHandler<1>::active_cell_iterator cell=dof_handler.begin_active(), endc=dof_handler.end();
        
	std::vector<double> sol_cell(n_q_points);
        
	std::vector< Point <1> > quad_points(n_q_points);
	double logz(0.);
        
	std::vector< Point<1> > vertices(dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(MappingQ1<1>(), dof_handler, vertices);
        
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
void LevyIntegralPrice<2>::compute_J(dealii::Vector<double> & sol, dealii::DoFHandler<2> & dof_handler, dealii::FE_Q<2> & fe) {
	using namespace dealii;
	using namespace std;
        
        
        const unsigned N(sol.size());
        
	j1.reinit(N);
        j2.reinit(N);
        
	QGauss<1> quad1D(3);    
	FEFaceValues<2> fe_face(fe, quad1D, update_values  | update_quadrature_points | update_JxW_values);
        
	const unsigned int n_q_points=quad1D.size();
        
	typename DoFHandler<2>::active_cell_iterator   endc=dof_handler.end();
        
	std::vector< Point<2> > vertices(dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_handler, vertices);
        
	double z, karg;
        
	typename DoFHandler<2>::active_cell_iterator cell=dof_handler.begin_active();
        
	std::vector<Point <2> > quad_points(n_q_points);
	std::vector<double> sol_values(n_q_points);
        
	Point<2> actual_vertex;
        for (;cell !=endc;++cell) {
                
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
		{	  
                        double center(cell->face(3)->center()(1));
                        fe_face.reinit(cell, 3);
                        
                        quad_points=fe_face.get_quadrature_points();
                        
                        fe_face.get_function_values(sol,  sol_values);
                        
                        for (unsigned it=0;it<N;++it) {
                                actual_vertex=vertices[it];
                                if (fabs(actual_vertex(1)-center)<constants::grid_toll)
                                {
                                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                                z=quad_points[q_point](0);
                                                karg=log(z/actual_vertex(0));
                                                j1[it]+=fe_face.JxW(q_point)*((*mods[0]).density(karg))*(sol_values[q_point])/z;
                                        }
                                }
                                
                        }
		}
		{	  
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
        
        //         ofstream out("JdiLevy", ios_base::app);
        //         out<< "j1 is \n"<< j1<<"\n\nj2 is \n"<< j2<< "\n";
        
}


# endif