#ifndef __quadrature_hpp
#define __quadrature_hpp

#include "QuadratureRule.hpp"

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

#endif