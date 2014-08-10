#include <iostream>
#include "Levy.hpp"

using namespace std;
using namespace dealii;

int main(){
	
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        EuropeanOptionPrice<2> mickey(OptionType::Put, model1.get_pointer(), model2.get_pointer(),
                                      -0.2, 0.1, 1., 200., 7, 100);
        
        EuropeanOptionLogPrice<2> mickey2(OptionType::Put, model1.get_pointer(), model2.get_pointer(),
                                          -0.2, 0.1, 1., 200., 7, 100);
        
        AmericanOptionPrice<2> mickey3(model1.get_pointer(), model2.get_pointer(),
                                       -0.2, 0.1, 1., 200., 7, 100);
        
        AmericanOptionLogPrice<2> mickey4(model1.get_pointer(), model2.get_pointer(),
                                          -0.2, 0.1, 1., 200., 7, 100);
        
        mickey.run();
        mickey2.run();
        mickey3.run();
        mickey4.run();
        
        cout<<mickey.get_price()<<"\n";
        cout<<mickey2.get_price()<<"\n";
        cout<<mickey3.get_price()<<"\n";
        cout<<mickey4.get_price()<<"\n";
        
        return 0;
}