#include <iostream>
#include "Factory.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        // We create here a BlackScholesModel object with spot price and volatility
        BlackScholesModel model(95., 0.120381);
        
        {
                // We create an object of type Option, a European Put with the Price transformation
                // with the model created above.
                auto foo=OptionFactory::instance()->create(ExerciseType::EU,
                                                           OptionType::Put,
                                                           Transformation::Price,
                                                           model.get_pointer(),
                                                           0.0367, 1., 90., 12, 250);
                
                foo->set_print(true);
                foo->set_timing(true);
                foo->run();
                
                auto times=foo->get_times();
                
                cout<<"The price of the option is "<<foo->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }
        
        {
                // We create an object of type Option, a European Put with the LogPrice transformation
                // with the model created above.
                auto goofy=OptionFactory::instance()->create(ExerciseType::EU,
                                                             OptionType::Put,
                                                             Transformation::LogPrice,
                                                             model.get_pointer(),
                                                             0.0367, 1., 90., 12, 250);
                
                goofy->set_print(true);
                goofy->set_timing(true);
                goofy->run();
                
                auto times=goofy->get_times();
                
                cout<<"The price of the option is "<<goofy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        // We create here two BlackScholesModel objects with spot price and volatility
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        {
                auto duffy=OptionFactory::instance()->create(ExerciseType::EU,
                                                             OptionType::Call,
                                                             Transformation::Price,
                                                             model1.get_pointer(),
                                                             model2.get_pointer(),
                                                             -0.2, 0.1, 1., 200., 7, 100);
                
                duffy->set_print_grid(true);
                duffy->set_print(true);
                duffy->set_timing(true);
                duffy->run();
                
                auto times=duffy->get_times();
                
                cout<<"The price of the option is "<<duffy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        {
                auto minnie=OptionFactory::instance()->create(ExerciseType::EU,
                                                             OptionType::Call,
                                                             Transformation::LogPrice,
                                                             model1.get_pointer(),
                                                             model2.get_pointer(),
                                                             -0.2, 0.1, 1., 200., 7, 100);
                
                minnie->set_print_grid(true);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        return 0;
        
}