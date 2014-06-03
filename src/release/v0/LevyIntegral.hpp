#ifndef __integral_levy_hpp
#define __integral_levy_hpp

#include "QuadratureRule.hpp"
#include "Densities.hpp"

// classe astratta

template <unsigned dim>
class LevyIntegral {
private:
        
        
public:
        virtual double  get_part1();
        virtual void    get_part2(dealii::Vector<double> &J);
};

// classe Kou -> calcola alpha/lambda, calcola J

// classe Merton -> calcola alpha/lambda, calcola J

#endif