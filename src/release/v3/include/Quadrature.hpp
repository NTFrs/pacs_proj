#ifndef __quadrature_hpp
#define __quadrature_hpp

#include "QuadratureRule.hpp"
#include "Constants.hpp"

//! Laguerre Nodes for Kou model
/*!
 * This class is used to compute the Laguerre Nodes needed in Kou model's integration step.
 */
class Quadrature_Laguerre{
private:
        
        std::vector<double> nodes;
        std::vector<double> weights;
        unsigned order; 
        
public:
        //! Sintetic constructor
        Quadrature_Laguerre()=default;
        
        //! Constructor
        /*!
         * The constructor needs the order of the methods (i.e. the number of integration nodes) and the parameter of the exponential.
         */
        Quadrature_Laguerre(unsigned n, double lambda){
                
                order=n;
                
                nodes=std::vector<double> (order);
                weights=std::vector<double> (order);
                
                unsigned kind = 5; // kind=5, Generalized Laguerre, (a,+oo) (x-a)^alpha*exp(-b*(x-a))
                
                quadrature::cgqf ( order, kind, 0., 0., 0., lambda, nodes.data(), weights.data() );
        }
        
        //! Returns the order of the method
        inline const unsigned get_order () {return order;}
        //! Returns the integration nodes
        inline std::vector<double> const & get_nodes () {return nodes;}
        //! Returns the integration weights
        inline std::vector<double> const & get_weights () {return weights;}
};


//! Hermite Nodes for Kou model
/*!
 * This class is used to compute the Gauss-Hermite Nodes needed in Merton model's integration step.
 */
class Quadrature_Hermite{
private:
        
	std::vector<double> nodes;
	std::vector<double> weights;
	unsigned order; 
        
public:
        
	Quadrature_Hermite()=default;
        
	//! Constructor
	/*!
	 * The constructor needs the order of the methods (i.e. the number of integration nodes) and the parameter of the gaussian.
	 */
	Quadrature_Hermite(unsigned n, double mu, double delta){
                
                order=n;
                
                nodes=std::vector<double> (order);
                weights=std::vector<double> (order);
                
                unsigned kind = 6;                          // kind=6, Generalized Hermite, (-inf,inf)  |x-a|^alpha*exp(-b*(x-a)^2)
                
                //cgqf ( int nt, int kind, double alpha, double beta, double a, double b, double t[], double wts[] )
                quadrature::cgqf ( order, kind, 0., 0., mu, 1/(2*delta*delta), nodes.data(), weights.data() );
        }
    
    //! Returns the order of the method
	inline unsigned get_order () {return order;}
	//! Returns the integration nodes
	inline std::vector<double> const & get_nodes () {return nodes;}
	//! Returns the integration weights
	inline std::vector<double> const & get_weights () {return weights;}
};


#endif