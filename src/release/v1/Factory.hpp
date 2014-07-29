#ifndef __factory_hpp
#define __factory_hpp

#include "Levy.hpp"

class OptionFactory {
private:
        
        OptionFactory(){};
        OptionFactory(const OptionFactory &)=delete;
        OptionFactory &operator=(const OptionFactory &) { return *this; }
        
public:
        static OptionFactory * get()
        {
                static OptionFactory instance;
                return &instance;
        }
        
        // 1d creator
        std::unique_ptr<OptionBase<1> > create(ExerciseType type1,
                                               OptionType type2,
                                               Transformation t,
                                               Model * const m,
                                               double r,
                                               double T,
                                               double K,
                                               unsigned N,
                                               unsigned M) {
                if (type1==ExerciseType::EU) {
                        if (t==Transformation::Price) {
                                return std::unique_ptr<OptionBase<1> >
                                (new EuropeanOptionPrice<1>(type2, m, r, T, K, N, M));
                        }
                        else
                                return std::unique_ptr<OptionBase<1> >
                                (new EuropeanOptionLogPrice<1>(type2, m, r, T, K, N, M));
                }
                else {
                        if (type2!=OptionType::Put) {
                                throw(std::logic_error("Error! American options MUST be Put.\n"));
                        }
                        else {
                                if (t==Transformation::Price) {
                                        return std::unique_ptr<OptionBase<1> >
                                        (new AmericanOptionPrice<1>(m, r, T, K, N, M));
                                }
                                else
                                        return std::unique_ptr<OptionBase<1> >
                                        (new AmericanOptionLogPrice<1>(m, r, T, K, N, M));
                        }
                }
        }
        
        // 2d creator
        std::unique_ptr<OptionBase<2> > create(ExerciseType type1,
                                               OptionType type2,
                                               Transformation t,
                                               Model * const m1,
                                               Model * const m2,
                                               double rho,
                                               double r,
                                               double T,
                                               double K,
                                               unsigned N,
                                               unsigned M) {
                if (type1==ExerciseType::EU) {
                        if (t==Transformation::Price) {
                                return std::unique_ptr<OptionBase<2> >
                                (new EuropeanOptionPrice<2>(type2, m1, m2, rho, r, T, K, N, M));
                        }
                        else
                                return std::unique_ptr<OptionBase<2> >
                                (new EuropeanOptionLogPrice<2>(type2, m1, m2, rho, r, T, K, N, M));
                }
                else {
                        if (type2!=OptionType::Put) {
                                throw(std::logic_error("Error! American options MUST be Put.\n"));
                        }
                        else {
                                if (t==Transformation::Price) {
                                        return std::unique_ptr<OptionBase<2> >
                                        (new AmericanOptionPrice<2>(m1, m2, rho, r, T, K, N, M));
                                }
                                else
                                        return std::unique_ptr<OptionBase<2> >
                                        (new AmericanOptionLogPrice<2>(m1, m2, rho, r, T, K, N, M));
                        }
                }
        }
        
        
};

#endif