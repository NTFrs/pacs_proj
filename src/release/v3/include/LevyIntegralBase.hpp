#ifndef __integral_levy_base_hpp
#define __integral_levy_base_hpp

#include <cmath>
#include <memory>

#include "dealii.hpp"
#include "Quadrature.hpp"
#include "Tools.hpp"
#include "Models.hpp"
#include "Constants.hpp"

//! Abstract class for the Levy Classes which handle the integral part
/*!
 * This class gives the base for the classes calculating the integral part of the equation. It also implements some basic methods that all children classes will have.
 */

template<unsigned dim>
class LevyIntegralBase{
protected:
	
	std::vector<double> alpha;
	dealii::Vector<double> j1, j2;
	
	
	dealii::Point<dim> bMin, bMax, lower_limit, upper_limit;
	
	//should be const? 
	std::vector<Model *> mods;
	
	bool alpha_ran;
	bool j_ran;
	
	//! Computes the alpha part of both integrals
	/*!
         * Computes the alpha part of the integrals using the generic Gauss quadrature nodes, thus working with any model.
         */
	virtual void compute_alpha();
	//! Computes the bounds needed to compute the integral with Gauss nodes.
	virtual void compute_bounds();
	
public:
        virtual ~LevyIntegralBase()=default;
        
	LevyIntegralBase()=delete;
        
        LevyIntegralBase(const LevyIntegralBase &)=delete;
        
        //! Constructor
        /*!
         *  This is the only provided constructor. Since this is an abstract class,  it only serves when called by classes derived from this one.
         * \param lower_limit_ 		the left-bottom limit of the domain		
         * \param upper_limit_ 		the rigth-upper limit of the domain
         * \param Models_			A vector containing the needed models
         */
	LevyIntegralBase(dealii::Point<dim> lower_limit_,
                         dealii::Point<dim> upper_limit_,
                         std::vector<Model *> & Models_)
        :
        lower_limit(lower_limit_),
        upper_limit(upper_limit_),
        mods(Models_),
        alpha_ran(false),
        j_ran(false)
        {
                if (Models_.size() !=dim)
                        //TODO add exception
                        std::cerr<< "Wrong dimension! Number of models is different from option dimension\n";
                this->compute_bounds();
        };
	
	LevyIntegralBase& operator=(const LevyIntegralBase &)=delete;
	
	virtual void compute_J(dealii::Vector<double> & sol, 
                               dealii::DoFHandler<dim> & dof_handler,
	 dealii::FE_Q<dim> & fe, std::vector< dealii::Point<dim> > const & vertices)=0;
	
	virtual inline void set_time(double tm){};
        
        virtual void set_adaptivity_params(unsigned order_max_, double alpha_toll_, double J_toll_){};
        
        virtual void set_adaptivity_params(unsigned order_max_, double alpha_toll_){};
	
	//! Returns the value of the alpha part of the first integral
	virtual double get_alpha_1() {
                if (!alpha_ran) {
                        this->compute_alpha();
                }
                return alpha[0];
        }
	//! Returns the value of the alpha part of the second integral
	virtual double get_alpha_2() {
                //TODO add exception
                if (dim<2) {
                        std::cerr<< "Dimension of this object is 1,  can't calculate alpha_2\n";
                        exit(-1);
                }
                if (!alpha_ran) {                    // add exception if dim < 2
                        this->compute_alpha();
                }
                return alpha[1];
        }
        //! Fills a vector with the values of the alpha part
	virtual void get_alpha(std::vector<double> & alp) {
                if (!alpha_ran) {
                        this->compute_alpha();
                }
                alp=alpha;
                return;
        }
	
	//! Used to get the j part of the first integral
	virtual void get_j_1(dealii::Vector<double> * &J_x) {
                if (!j_ran) {
                        //TODO add exception
                        std::cerr<< "Run J first!"<< std::endl;
                }
                else {
                        J_x=&j1;
                        j_ran=false;
                }
                return;
        }
        
        //! Used to get j parts of both integrals
	virtual void get_j_both(dealii::Vector<double> * &J_x, dealii::Vector<double> * &J_y) {
                if (!j_ran) {
                        std::cerr<< "Run J first!"<< std::endl;
                }
                else{
                        J_x=&j1;
                        J_y=&j2;
                        j_ran=false;
                }
                return;
        }
	
};

template<unsigned dim>
void LevyIntegralBase<dim>::compute_bounds() {
	for (unsigned d=0;d<dim;++d) {
                //TODO may misbehave with merton model,  but Merton has a special quadrature so no big deal
                bMin[d]=std::min(0., lower_limit(d));
                bMax[d]=upper_limit(d);
                while ((*mods[d]).density(bMin[d])>constants::light_toll)
                        bMin[d]+=-0.5;
                while ((*mods[d]).density(bMax[d])>constants::light_toll)
                        bMax[d]+=0.5;
                
        }
}

template<unsigned dim>
void LevyIntegralBase<dim>::compute_alpha()
{
	using namespace dealii;
	alpha=std::vector<double>(dim, 0.);
        //an alpha is computed for each dimension
        for (unsigned d=0;d<dim;++d) {
                
                //a one dimensional grid is built,  along with the finite elements
                Triangulation<1> integral_grid;
                FE_Q<1> fe2(1);
                DoFHandler<1> dof_handler2(integral_grid);
                
                GridGenerator::hyper_cube<1>(integral_grid, bMin[d], bMax[d]);
                integral_grid.refine_global(15);
                
                dof_handler2.distribute_dofs(fe2);
                
                //and we use a stadard gauss quadrature
                QGauss<1> quadrature_formula(8);
                FEValues<1> fe_values(fe2, quadrature_formula,  update_quadrature_points |update_JxW_values);
                
                typename DoFHandler<1>::active_cell_iterator
                cell=dof_handler2.begin_active(),
                endc=dof_handler2.end();
                
                const unsigned int n_q_points(quadrature_formula.size());
                
                //we thus loop over all cells
                for (; cell !=endc;++cell) {
                        
                        fe_values.reinit(cell);
                        std::vector< Point<1> > quad_points_1D(fe_values.get_quadrature_points());
                        
                        //in each cell we compute the contribution to alpha[d]. fe_values.JxW is the jacobian (respect to a reference cell) and the quadrature weight provided by dealii.
                        for (unsigned q_point=0;q_point<n_q_points;++q_point) {
                                double p;
                                p=quad_points_1D[q_point][0];
                                alpha[d]+=fe_values.JxW(q_point)*(exp(p)-1.)*(*mods[d]).density(p);
                        }
                }
        }
	alpha_ran=true;
}





#endif