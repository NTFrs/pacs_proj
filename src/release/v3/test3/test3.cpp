#include <iostream>
#include "Factory.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        cout<<"*** American Put Price 1d, Black&Scholes\n";
        /*
        BlackScholesModel model(95., 0.120381);
        
        {
                auto foo=Factory::instance()->create(ExerciseType::US,
                                                     OptionType::Put,
                                                     Transformation::Price,
                                                     model.get_pointer(),
                                                     0.0367, 1., 90., 10, 100);
                
                foo->set_print(true);
                foo->set_timing(true);
                foo->run();
                
                auto times=foo->get_times();
                
                cout<<"The price of the option is "<<foo->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** American Put LogPrice 1d, Black&Scholes\n";
        
        {
                auto goofy=Factory::instance()->create(ExerciseType::US,
                                                       OptionType::Put,
                                                       Transformation::LogPrice,
                                                       model.get_pointer(),
                                                       0.0367, 1., 90., 10, 100);
                
                goofy->set_print(true);
                goofy->set_timing(true);
                goofy->run();
                
                auto times=goofy->get_times();
                
                cout<<"The price of the option is "<<goofy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** American Put Price 2d, Black&Scholes\n";
        
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        
        {
                auto duffy=Factory::instance()->create(ExerciseType::US,
                                                       OptionType::Put,
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
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** American Put LogPrice 2d, Black&Scholes\n";
        
        {
                auto mickey=Factory::instance()->create(ExerciseType::US,
                                                        OptionType::Put,
                                                        Transformation::LogPrice,
                                                        model1.get_pointer(),
                                                        model2.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 7, 100);
                
                mickey->set_print_grid(true);
                mickey->set_print(true);
                mickey->set_timing(true);
                mickey->run();
                
                auto times=mickey->get_times();
                
                cout<<"The price of the option is "<<mickey->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        */
        cout<<"*** American Put Price 1d, Kou\n";
        
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        
        {
                auto minnie=Factory::instance()->create(ExerciseType::US,
                                                        OptionType::Put,
                                                        Transformation::Price,
                                                        model3.get_pointer(),
                                                        0.0367, 1., 90., 10, 100);
                
                minnie->set_scale_factor(0.99);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** American Put LogPrice 1d, Kou\n";
        
        {
                auto daisy=Factory::instance()->create(ExerciseType::US,
                                                       OptionType::Put,
                                                       Transformation::LogPrice,
                                                       model3.get_pointer(),
                                                       0.0367, 1., 90., 10, 100);
                
                daisy->set_print(true);
                daisy->set_timing(true);
                daisy->run();
                
                auto times=daisy->get_times();
                
                cout<<"The price of the option is "<<daisy->get_price()<<", evaluated in "<<times.second/1.e6<<".\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        return 0;
}
