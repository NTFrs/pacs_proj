#include <iostream>
#include "Levy.hpp"

int main()
{
        
        using namespace dealii;
        using namespace std;
        
        cout<<"*** European Put 1d.\n";
        
        // We create here a BlackScholesModel object with spot price and volatility
        BlackScholesModel model(95., 0.120381);
        
        {
                // We create an object of type Option, a European Put with the Price transformation
                // with the model created above.
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Put,
                                                     Transformation::LogPrice,
                                                     model.get_pointer(),
                                                     0.0367, 1., 90., 4, 5);
                
                foo->set_print(true);
                foo->set_timing(true);
                foo->run();
                
                auto times=foo->get_times();
                
        }
         
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        
        {
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Call,
                                                     Transformation::LogPrice,
                                                     model3.get_pointer(),
                                                     0.0367, 1., 90., 4, 5);
                
                foo->set_print(true);
                foo->set_timing(true);
                foo->run();
                
                auto times=foo->get_times();
                
                cout<<"The price of the option is "<<foo->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }
        
        MertonModel model1(80., 0.2, 0.1, 0.4552, 0.258147);
        MertonModel model2(120., 0.1, -0.390078, 0.338796, 0.174814);
        
        {
                auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Call,
                                                        Transformation::Price,
                                                        model1.get_pointer(),
                                                        model2.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 2, 5);
                
                minnie->set_print_grid(true);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        return 0;
}
