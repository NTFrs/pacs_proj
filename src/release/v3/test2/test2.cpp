#include <iostream>
#include "Levy.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        // We test here a 1d PIDE in both transformation.
        
        KouModel model(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        
        {
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Call,
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
        
        {
                auto goofy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Call,
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
        
        // We test here the convergence of the price modifing the width of the grid.
        
        {
                std::array<double, 4>      f={0., 0.33, 0.66, 0.99};       // scale factor
                std::array<double, 4>      p={0., 0., 0., 0.};             // prices from price transformation
                std::array<double, 4>      l={0., 0., 0., 0.};             // prices from logprice transformation
                
                auto duffy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Put,
                                                       Transformation::Price,
                                                       model.get_pointer(),
                                                       0.0367, 1., 90., 10, 100);
                
                auto mickey=Factory::instance()->create(ExerciseType::EU,
                                                        OptionType::Put,
                                                        Transformation::LogPrice,
                                                        model.get_pointer(),
                                                        0.0367, 1., 90., 10, 100);
                
                // we ask the object to avoid any print
                duffy->set_verbose(false);
                mickey->set_verbose(false);
                
                for (unsigned i=0; i<f.size(); ++i) {
                        // we reset the options
                        duffy->reset();
                        mickey->reset();
                        
                        // we set the scale factor
                        duffy->set_scale_factor(f[i]);
                        mickey->set_scale_factor(f[i]);
                        
                        cout<<"Evaluting Price with f="<<f[i]<<"...\n";
                        p[i]=duffy->get_price();
                        cout<<"Evaluting LogPrice with f="<<f[i]<<"...\n";
                        l[i]=mickey->get_price();
                }
                
                cout<<"Convergence table varying the scale factor.\nf:\t";
                for (unsigned i=0; i<f.size(); ++i) {
                        cout<<f[i]<<"\t";
                }
                cout<<"\np:\t";
                for (unsigned i=0; i<f.size(); ++i) {
                        cout<<p[i]<<"\t";
                }
                cout<<"\nl:\t";
                for (unsigned i=0; i<f.size(); ++i) {
                        cout<<l[i]<<"\t";
                }
                cout<<"\n";
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        // We test here a 2d call with Merton model
        
        MertonModel model1(80., 0.2, 0.1, 0.4552, 0.258147);
        MertonModel model2(120., 0.1, -0.390078, 0.338796, 0.174814);
        
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
                daisy->run();
                
                auto times=daisy->get_times();
                
                cout<<"The price of the option is "<<daisy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** Do you want to perform some convergence tests for the integral part? (y/n) ";
        string s="y";
        //cin>>s;
        
        if (s=="y") {
                
                array<unsigned, 6>      max_order={4, 8, 16, 32, 64, 128};
                array<double, 6>        prices, times;
                
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Call,
                                                     Transformation::LogPrice,
                                                     model.get_pointer(),
                                                     0.0367, 1., 90., 10, 100);
                
                foo->set_timing(true);
                foo->set_verbose(false);
                
                for (unsigned i=0; i<max_order.size(); ++i) {
                        
                        cout<<"Evaluating option with "<<max_order[i]<<"...\n";
                        
                        foo->reset();
                        foo->set_integral_adaptivity_params(true, 4, max_order[i]);
                        
                        foo->run();
                        
                        auto time=foo->get_times();
                        prices[i]=foo->get_price();
                        times[i]=time.second;
                }
                cout<<"Convergence table\n";
                for (unsigned i=0; i<max_order.size(); ++i) {
                        cout<<"\t"<<max_order[i]<<"\t";
                }
                cout<<"\nPrices:";
                for (unsigned i=0; i<max_order.size(); ++i) {
                        cout<<"\t"<<prices[i]<<"\t";
                }
                cout<<"\nTimes:";
                for (unsigned i=0; i<max_order.size(); ++i) {
                        cout<<"\t"<<times[i]/1.e6<<"s\t";
                }
                cout<<"\n";
                
        }
        else if (s!="n") {
                throw(logic_error("Something went wrong..."));
        }
        
        return 0;
}
