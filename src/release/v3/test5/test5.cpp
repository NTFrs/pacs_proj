#include <iostream>
#include "Levy.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        // We test here a 1d PIDE in both transformation with the mesh adaptivity.
        
        cout<<"*** Mesh Adaptivity, Kou 1d\n";
        
        KouModel model(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        
        {
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Call,
                                                     Transformation::Price,
                                                     model.get_pointer(),
                                                     0.0367, 1., 90., 8, 100);
                
                foo->set_print(true);
                foo->set_timing(true);
                foo->set_refine_status(true, 0.2, 0.03);
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
                                                       0.0367, 1., 90., 8, 100);
                
                goofy->set_print(true);
                goofy->set_timing(true);
                goofy->set_refine_status(true, 0.2, 0.03);
                goofy->run();
                
                auto times=goofy->get_times();
                
                cout<<"The price of the option is "<<goofy->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
        }
        
        cout<<"*** Mesh Adaptivity, American Option 1d, B&S\n";
        
        BlackScholesModel model1(95., 0.120381);
        
        {
                auto duffy=Factory::instance()->create(ExerciseType::US,
                                                       OptionType::Put,
                                                       Transformation::Price,
                                                       model1.get_pointer(),
                                                       0.0367, 1., 90., 10, 100);
                
                duffy->set_timing(true);
                duffy->set_verbose(false);
                duffy->set_scale_factor(0.99);
                duffy->set_refine_status(true, 0.2, 0.);
                
                array<unsigned, 3>      maxiter={100, 1000, 10000};
                array<double, 3>        prices;
                array<double, 3>        time;
                
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        duffy->reset();
                        duffy->set_PSOR_parameters(constants::high_toll, maxiter[i]);
                        duffy->run();
                        auto times=duffy->get_times();
                        prices[i]=duffy->get_price();
                        time[i]=times.second;
                }
                
                cout<<"maxiter:\t";
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        cout<<maxiter[i]<<"\t";
                }
                cout<<"\nprice:\t";
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        cout<<prices[i]<<"\t";
                }
                cout<<"\ntime:\t";
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        cout<<time[i]<<"s\t";
                }
                cout<<"\n";
                
        }
        
        {
                auto duffy=Factory::instance()->create(ExerciseType::US,
                                                       OptionType::Put,
                                                       Transformation::LogPrice,
                                                       model1.get_pointer(),
                                                       0.0367, 1., 90., 10, 100);
                
                duffy->set_timing(true);
                duffy->set_verbose(false);
                duffy->set_scale_factor(0.99);
                duffy->set_refine_status(true, 0.5, 0.);
                
                array<unsigned, 3>      maxiter={100, 1000, 10000};
                array<double, 3>        prices;
                array<double, 3>        time;
                
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        duffy->reset();
                        duffy->set_PSOR_parameters(constants::high_toll, maxiter[i]);
                        duffy->run();
                        auto times=duffy->get_times();
                        prices[i]=duffy->get_price();
                        time[i]=times.second;
                }
                
                cout<<"maxiter:\t";
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        cout<<maxiter[i]<<"\t";
                }
                cout<<"\nprice:\t";
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        cout<<prices[i]<<"\t";
                }
                cout<<"\ntime:\t";
                for (unsigned i=0; i<maxiter.size(); ++i) {
                        cout<<time[i]<<"s\t";
                }
                cout<<"\n";
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        cout<<"*** Mesh Adaptivity, Merton 2d\n";
        
        MertonModel model2(80., 0.2, 0.1, 0.4552, 0.258147);
        MertonModel model3(120., 0.1, -0.390078, 0.338796, 0.174814);
        
        {
                {
                        auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                                OptionType::Call,
                                                                Transformation::LogPrice,
                                                                model2.get_pointer(),
                                                                model3.get_pointer(),
                                                                -0.2, 0.1, 1., 200., 6, 100);
                        
                        minnie->set_print_grid(true);
                        minnie->set_print(true);
                        minnie->set_timing(true);
                        minnie->set_integral_adaptivity_params(false);
                        
                        minnie->set_refine_status(true, 0.03, 0.15);
                        minnie->run();
                        
                        auto times=minnie->get_times();
                        
                        cout<<"Refinement=0.03, Coarsening=0.15.\nThe price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                }
                cout<<"Press return to continue...\n";
                //cin.get();
                
                {
                        auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                                OptionType::Call,
                                                                Transformation::LogPrice,
                                                                model2.get_pointer(),
                                                                model3.get_pointer(),
                                                                -0.2, 0.1, 1., 200., 6, 100);
                        
                        minnie->set_print_grid(true);
                        minnie->set_print(true);
                        minnie->set_timing(true);
                        minnie->set_integral_adaptivity_params(false);
                        
                        minnie->set_refine_status(true, 0., 0.1);
                        minnie->run();
                        
                        auto times=minnie->get_times();
                        
                        cout<<"Refinement=0., Coarsening=0.1.\nThe price of the option is "<<minnie->get_price()<<", evaluated in "<<times.second/1.e6<<"s.\n";
                }
                
        }
        
        cout<<"Press return to continue...\n";
        //cin.get();
        
        return 0;
}