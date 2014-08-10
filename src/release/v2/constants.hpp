#ifndef __constants_hpp
#define __constants_hpp

#include <climits>

//! We gather here some of the constants used in the whole program.
namespace constants {
        const double pi=4.*std::atan(1.);
        const double grid_toll=1.e-5;
        const double light_toll=1.e-8;
        const double high_toll=1.e-12;
        const double eps=std::numeric_limits<double>::epsilon();
}

#endif