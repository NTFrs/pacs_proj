#ifndef __factory_hpp
#define __factory_hpp

#include "Levy.hpp"

//! This class is a factory to instantiate Option-type objects
/*! This class uses a function, create, to instantiate objects of every Option-type: european and american, put and call, Price and LogPrice, 1d and 2d.
 */

class OptionFactory {
private:
        
        OptionFactory(){};
        OptionFactory(const OptionFactory &)=delete;
        OptionFactory &operator=(const OptionFactory &) { return *this; }
        
public:
        static OptionFactory * instance()
        {
                static OptionFactory instance;
                return &instance;
        }
        
        //! 1d creator
        /*! This function is used to create 1d Options.
         * \param type1         European or American
         * \param type2         Put or Call (note: American Options MUST be Put)
         * \param t             Transformation type: Price or LogPrice
         * \param m             Model class pointer
         * \param r             Interest rate
         * \param T             Time to Maturity
         * \param K             Strike Price
         * \param N             Refinement of the grid (e.g. insert 10 for 2^10=1024 cells)
         * \param M             Number of TimeStep
         */
        std::unique_ptr< OptionBase<1> > create(ExerciseType type1,
                                                OptionType type2,
                                                Transformation t,
                                                Model * const m,
                                                double r,
                                                double T,
                                                double K,
                                                unsigned N,
                                                unsigned M);
        
        //! 2d creator
        /*! This function is used to create 2d Options.
         * \param type1         European or American
         * \param type2         Put or Call (note: American Options MUST be Put)
         * \param t             Transformation type: Price or LogPrice
         * \param m1            First Model class pointer
         * \param m2            Second Model class pointer
         * \param rho           Correlation Index
         * \param r             Interest rate
         * \param T             Time to Maturity
         * \param K             Strike Price
         * \param N             Refinement of the grid (e.g. insert 10 for 2^10=1024 cells)
         * \param M             Number of TimeStep
         */
        std::unique_ptr< OptionBase<2> > create(ExerciseType type1,
                                                OptionType type2,
                                                Transformation t,
                                                Model * const m1,
                                                Model * const m2,
                                                double rho,
                                                double r,
                                                double T,
                                                double K,
                                                unsigned N,
                                                unsigned M);
        
};

std::unique_ptr< OptionBase<1> > OptionFactory::create(ExerciseType type1,
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

std::unique_ptr< OptionBase<2> > OptionFactory::create(ExerciseType type1,
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

#endif