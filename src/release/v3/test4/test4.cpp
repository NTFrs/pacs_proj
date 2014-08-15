#include <iostream>
#include "Factory.hpp"

int main(){
	
        using namespace dealii;
        using namespace std;
        
        cout<<"*** SpeedTest for LogPrice 1d, serial vs. parallel.\n";
        
        unsigned number_of_threads=omp_get_max_threads();
        
        KouModel model(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        
        array<unsigned, 4>         refinement={8, 9, 10, 11};
        array<double, 4>           times_s={0., 0., 0., 0.};
        array<double, 4>           times_p={0., 0., 0., 0.};
        array<double, 4>           price_s={0., 0., 0., 0.};
        array<double, 4>           price_p={0., 0., 0., 0.};
        
        // Serial
        {
                omp_set_num_threads(1);
                
                auto foo=Factory::instance()->create(ExerciseType::EU,
                                                     OptionType::Call,
                                                     Transformation::LogPrice,
                                                     model.get_pointer(),
                                                     0.0367, 1., 90., 1, 100);
                
                foo->set_verbose(false);
                
                for (unsigned i=0; i<refinement.size(); ++i) {
                        foo->reset();
                        foo->set_refs(refinement[i]);
                        foo->set_timing(true);
                        
                        foo->run();
                        
                        price_s[i]=foo->get_price();
                        
                        auto clock=foo->get_times();
                        times_s[i]=clock.second;
                }
        }
        //Parallel
        {
                omp_set_num_threads(number_of_threads);
                
                auto goofy=Factory::instance()->create(ExerciseType::EU,
                                                       OptionType::Call,
                                                       Transformation::LogPrice,
                                                       model.get_pointer(),
                                                       0.0367, 1., 90., 1, 100);
                
                goofy->set_verbose(false);
                
                for (unsigned i=0; i<refinement.size(); ++i) {
                        goofy->reset();
                        goofy->set_refs(refinement[i]);
                        goofy->set_timing(true);
                        
                        goofy->run();
                        
                        price_p[i]=goofy->get_price();
                        
                        auto clock=goofy->get_times();
                        times_p[i]=clock.second;
                }
        }
        
        cout<<"*** Times tables 1d (with "<<number_of_threads<<" threads).\nSerial:\n";
        cout<<"Grid\tPrice\tTime\tRatio\n";
        for (unsigned i=0; i<refinement.size(); ++i) {
                cout<<pow(2, refinement[i])<<"\t"<<price_s[i]<<"\t"<<times_s[i]/1.e6<<"s\t";
                if (i>0) {
                        cout<<times_s[i]/times_s[i-1]<<"\t";
                }
                cout<<"\n";
        }
        cout<<"Parallel:\n";
        cout<<"Grid\tPrice\tTime\tRatio\tSpeedUp\n";
        for (unsigned i=0; i<refinement.size(); ++i) {
                cout<<pow(2, refinement[i])<<"\t"<<price_p[i]<<"\t"<<times_p[i]/1.e6<<"s\t";
                if (i>0) {
                        cout<<times_p[i]/times_p[i-1]<<"\t";
                }
                cout<<times_s[i]/times_p[i]<<"x\n";
        }
        
        cout<<"*** Do you want to perform also a 2d test? (y/n) ";
        string s;
        cin>>s;
        
        if (s=="y") {
                KouModel model1(80, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
                KouModel model2(120, 0.2, 0.33189, 0.400123, 8.3456, 5.12904);
                
                refinement={4, 5, 6, 7};
                
                // Serial
                {
                        omp_set_num_threads(1);
                        
                        auto minnie=Factory::instance()->create(ExerciseType::EU,
                                                                OptionType::Call,
                                                                Transformation::LogPrice,
                                                                model1.get_pointer(),
                                                                model2.get_pointer(),
                                                                0.2, 0.0367, 1., 200., 1, 100);
                        
                        minnie->set_verbose(false);
                        
                        for (unsigned i=0; i<refinement.size(); ++i) {
                                minnie->reset();
                                minnie->set_refs(refinement[i]);
                                minnie->set_timing(true);
                                
                                minnie->run();
                                
                                price_s[i]=minnie->get_price();
                                
                                auto clock=minnie->get_times();
                                times_s[i]=clock.second;
                        }
                }
                //Parallel
                {
                        omp_set_num_threads(number_of_threads);
                        
                        auto daisy=Factory::instance()->create(ExerciseType::EU,
                                                               OptionType::Call,
                                                               Transformation::LogPrice,
                                                               model1.get_pointer(),
                                                               model2.get_pointer(),
                                                               0.2, 0.0367, 1., 200., 1, 100);
                        
                        daisy->set_verbose(false);
                        
                        for (unsigned i=0; i<refinement.size(); ++i) {
                                daisy->reset();
                                daisy->set_refs(refinement[i]);
                                daisy->set_timing(true);
                                
                                daisy->run();
                                
                                price_p[i]=daisy->get_price();
                                
                                auto clock=daisy->get_times();
                                times_p[i]=clock.second;
                        }
                }
                
                cout<<"*** Times tables 2d (with "<<number_of_threads<<" threads).\nSerial:\n";
                cout<<"Grid\tPrice\tTime\tRatio\n";
                for (unsigned i=0; i<refinement.size(); ++i) {
                        cout<<pow(2, 2*refinement[i])<<"\t"<<price_s[i]<<"\t"<<times_s[i]/1.e6<<"s\t";
                        if (i>0) {
                                cout<<times_s[i]/times_s[i-1]<<"\t";
                        }
                        cout<<"\n";
                }
                cout<<"Parallel:\n";
                cout<<"Grid\tPrice\tTime\tRatio\tSpeedUp\n";
                for (unsigned i=0; i<refinement.size(); ++i) {
                        cout<<pow(2, 2*refinement[i])<<"\t"<<price_p[i]<<"\t"<<times_p[i]/1.e6<<"s\t";
                        if (i>0) {
                                cout<<times_p[i]/times_p[i-1]<<"\t";
                        }
                        cout<<times_s[i]/times_p[i]<<"x\n";
                }
                
                return 0;
        }
        else if (s!="n") {
                throw(logic_error("Something went wrong..."));
        }
        else {
                return 0;
        }
        
        
}
