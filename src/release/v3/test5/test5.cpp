#include <iostream>
#include "Factory.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        MertonModel model1(80., 0.2, 0.1, 0.4552, 0.258147);
        MertonModel model2(120., 0.1, -0.390078, 0.338796, 0.174814);
        
        cout<<"Merton...\n";
        
        {
                auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Call,
                                                        Transformation::Price,
                                                        model1.get_pointer(),
                                                        model2.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 6, 100);
                
                minnie->set_print_grid(true);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->set_verbose(false);
                minnie->set_scale_factor(0.99);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        {
                auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Call,
                                                       Transformation::LogPrice,
                                                       model1.get_pointer(),
                                                       model2.get_pointer(),
                                                       -0.2, 0.1, 1., 200., 6, 100);
                daisy->set_print_grid(true);
                daisy->set_print(true);
                daisy->set_timing(true);
                daisy->set_verbose(false);
                daisy->set_scale_factor(0.99);
                daisy->run();
                
                auto times=daisy->get_times();
                
                cout<<"The price of the option is "<<daisy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        {
                auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Put,
                                                        Transformation::Price,
                                                        model1.get_pointer(),
                                                        model2.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 6, 100);
                
                minnie->set_print_grid(true);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->set_verbose(false);
                minnie->set_scale_factor(0.99);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        {
                auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Put,
                                                       Transformation::LogPrice,
                                                       model1.get_pointer(),
                                                       model2.get_pointer(),
                                                       -0.2, 0.1, 1., 200., 6, 100);
                daisy->set_print_grid(true);
                daisy->set_print(true);
                daisy->set_timing(true);
                daisy->set_verbose(false);
                daisy->set_scale_factor(0.99);
                daisy->run();
                
                auto times=daisy->get_times();
                
                cout<<"The price of the option is "<<daisy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        KouModel model3(80, 0.1256, 0.20761, 0.330966, 9.65997, 3.13868);
        KouModel model4(120, 0.2, 0.20761, 0.330966, 9.65997, 3.13868);
        
        cout<<"Kou...\n";
        
        {
                auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Call,
                                                        Transformation::Price,
                                                        model3.get_pointer(),
                                                        model4.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 6, 100);
                
                minnie->set_print_grid(true);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->set_verbose(false);
                minnie->set_scale_factor(0.99);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        {
                auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Call,
                                                       Transformation::LogPrice,
                                                       model3.get_pointer(),
                                                       model4.get_pointer(),
                                                       -0.2, 0.1, 1., 200., 6, 100);
                daisy->set_print_grid(true);
                daisy->set_print(true);
                daisy->set_timing(true);
                daisy->set_verbose(false);
                daisy->set_scale_factor(0.99);
                daisy->run();
                
                auto times=daisy->get_times();
                
                cout<<"The price of the option is "<<daisy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        {
                auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Put,
                                                        Transformation::Price,
                                                        model3.get_pointer(),
                                                        model4.get_pointer(),
                                                        -0.2, 0.1, 1., 200., 6, 100);
                
                minnie->set_print_grid(true);
                minnie->set_print(true);
                minnie->set_timing(true);
                minnie->set_verbose(false);
                minnie->set_scale_factor(0.99);
                minnie->run();
                
                auto times=minnie->get_times();
                
                cout<<"The price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        {
                auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Put,
                                                       Transformation::LogPrice,
                                                       model3.get_pointer(),
                                                       model4.get_pointer(),
                                                       -0.2, 0.1, 1., 200., 6, 100);
                daisy->set_print_grid(true);
                daisy->set_print(true);
                daisy->set_timing(true);
                daisy->set_verbose(false);
                daisy->set_scale_factor(0.99);
                daisy->run();
                
                auto times=daisy->get_times();
                
                cout<<"The price of the option is "<<daisy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }

        return 0;
}
