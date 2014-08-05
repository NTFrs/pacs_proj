#ifndef __optionparameters_hpp
#define __optionparameters_hpp

template <unsigned dim>
class OptionParameters {
private:
        double S01;
        double S02;
        double K;
        double r;
        double rho;
        
        unsigned payoff_type;
public:
        //! Default Constructor
        OptionParameters()=default;
        
        //! 1d Constructor
        OptionParameters(double S0_, double K, double r_);
        
        //! 2d Constructor 
        OptionParameters(double S01_, double S02_, double K_, double r_, double rho_);
        
        //! 2d Constructor
        OptionParameters(double S01_, double S02_, double r_, double rho_);
        
        //! Returns the rate
        inline double get_rate() { return r; }
        
        //! Returns the strike
        inline double get_strike() {
                if (payoff_type!=2)
                        throw(std::logic_error("Error! This kind of option has no strike.\n"));
                else
                        return K;
        }
        
        //! Returns the spot price in 1d
        inline double get_spot() {
                if (payoff_type!=0)
                        throw(std::logic_error("Error! This function cannot be used in 2d options.\n"));
                else
                        return S01;
        }
        
        //! Returns the first spot in 2d
        inline double get_spot1() {
                if (payoff_type==0)
                        throw(std::logic_error("Error! This function cannot be used in 1d options.\n"));
                else
                        return S01;
        }
        
        //! Returns the first spot in 2d
        inline double get_spot2() {
                if (payoff_type==0)
                        throw(std::logic_error("Error! This function cannot be used in 1d options.\n"));
                else
                        return S02;
        }
        
        //! Returns the correlation index in 2d
        inline double get_rho() {
                if (payoff_type==0)
                        throw(std::logic_error("Error! This function cannot be used in 1d options.\n"));
                else
                        return rho;
        }
        
        
};

template <>
OptionParameters<1>::OptionParameters(double S0_, double K_, double r_)
:
S01(S0_),
S02(-1.),
K(K_),
r(r_),
rho(-2.),
payoff_type(0)
{}

template <>
OptionParameters<2>::OptionParameters(double S01_, double S02_, double K_, double r_, double rho_)
:
S01(S01_),
S02(S02_),
K(K_),
r(r_),
rho(rho_),
payoff_type(1)
{}

template <>
OptionParameters<2>::OptionParameters(double S01_, double S02_, double r_, double rho_)
:
S01(S01_),
S02(S02_),
K(-1.),
r(r_),
rho(rho_),
payoff_type(2)
{}

#endif