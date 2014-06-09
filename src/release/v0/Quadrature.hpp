#ifndef __quadrature_hpp
#define __quadrature_hpp

#include "QuadratureRule.hpp"

// Laguerre Nodes for Kou model
class Quadrature_Laguerre{
private:
        
        std::vector<double> nodes;
        std::vector<double> weights;
        unsigned order; 
        
public:
        
        Quadrature_Laguerre()=default;
        
        // the constructor builds nodes and weights
        Quadrature_Laguerre(unsigned n, double lambda){
                
                order=n;
                
                nodes=std::vector<double> (order);
                weights=std::vector<double> (order);
                
                unsigned kind = 5; // kind=5, Generalized Laguerre, (a,+oo) (x-a)^alpha*exp(-b*(x-a))
                
                quadrature::cgqf ( order, kind, 0., 0., 0., lambda, nodes.data(), weights.data() );
        }
        
        inline const unsigned get_order () {return order;}
        inline std::vector<double> const & get_nodes () {return nodes;}
        inline std::vector<double> const & get_weights () {return weights;}
};

#endif