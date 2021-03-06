#include <iostream>
//#include "Levy.hpp"
#include "Factory.hpp"

/*
using namespace std;
using namespace dealii;
*/
int main(){
	
        using namespace dealii;
        using namespace std;
        
        BlackScholesModel model(95., 0.120381);
        
        BlackScholesModel model1(80., 0.1256);
        BlackScholesModel model2(120., 0.2);
        /*
        std::vector< reference_wrapper<Model> > v;
        v.push_back(model);
        v.push_back(model1);
        v.push_back(model2);
        
        cout<<v[1].get().get_spot()<<"\n";
        */
        KouModel model3(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);
        KouModel model4(80, 0.1256, 0.20761, 0.330966, 9.65997, 3.13868);
        
        KouModel model5(120, 0.2, 0.20761, 0.330966, 9.65997, 3.13868);
        
        MertonModel model6(80., 0.2, -0.390078, 0.338796, 0.174814);
        MertonModel model7(120., 0.2, -0.390078, 0.338796, 0.174814);
        
        OptionType tipo(OptionType::Call);
        
        EuropeanOptionPrice<1> a(tipo, model.get_pointer(), 0.0367, 1., 90., 9, 100);
        EuropeanOptionLogPrice<1> b(tipo, model.get_pointer(), 0.0367, 1., 90., 9, 100);
        
        EuropeanOptionPrice<2> c
        (tipo, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 100);
        
        EuropeanOptionLogPrice<2> d
        (tipo, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 100);
        
        a.set_scale_factor(0.5);
        b.set_scale_factor(0.5);
        c.set_scale_factor(0.5);
        d.set_scale_factor(0.5);
        
        a.set_timing(true);
        b.set_timing(true);
        c.set_timing(true);
        d.set_timing(true);
        
        a.set_print(true);
        b.set_print(true);
        c.set_print(true);
        d.set_print(true);
        //a.set_print_grid(true);
        //b.set_print_grid(true);
        c.set_print_grid(true);
        d.set_print_grid(true);
//         a.set_refine_status(true);
//         b.set_refine_status(true);
        //c.set_refine_status(true, 0.1);		
        //         d.set_refine_status(true);
        
        a.run();
        b.run();
        c.run();
        d.run();
        //c.print_grid("Griglia");
        //c.print_solution_gnuplot("Soluzione");
        
        cout<<a.get_price()<<"\n";
        cout<<b.get_price()<<"\n";
        cout<<c.get_price()<<"\n";
        cout<<d.get_price()<<"\n";
        
        auto x=a.get_times();
        cout<<x.first/1.e6<<" "<<x.second/1.e6<<"\n";
        auto y=b.get_times();
        cout<<y.first/1.e6<<" "<<y.second/1.e6<<"\n";
        auto z=c.get_times();
        cout<<z.first/1.e6<<" "<<z.second/1.e6<<"\n";
        auto w=d.get_times();
        cout<<w.first/1.e6<<" "<<w.second/1.e6<<"\n";
        
//         cout<<d.get_price()<<"\n";
        
	/*
         const int top=7-3+1;
         double T[2][top], result[2][top], real_T[2][top];
         
         clock_t inizio,fine;
         struct timeval start, end;
         
         for (int i=0; i<top; i++) {
         
         {
         EuropeanOptionPrice<2> c
         (OptionType::Call, model4.get_pointer(), model5.get_pointer(),
         -0.2, 0.1, 1., 200., i+3, 50);
         
         gettimeofday(&start, NULL);
         inizio=clock();
         c.run();
         gettimeofday(&end, NULL);
         fine=clock();
         
         result[0][i]=c.get_price();
         
         T[0][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
         real_T[0][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
         end.tv_usec - start.tv_usec);
         
         }
         
         {
         EuropeanOptionLogPrice<2> d
	 (OptionType::Call, model4.get_pointer(), model5.get_pointer(),
         -0.2, 0.1, 1., 200., i+3, 50);
         
         gettimeofday(&start, NULL);
         inizio=clock();
         d.run();
         gettimeofday(&end, NULL);
         fine=clock();
         
         result[1][i]=d.get_price();
         
         T[1][i]=static_cast<double> (((fine-inizio)*1.e6)/CLOCKS_PER_SEC);
         real_T[1][i]=((end.tv_sec  - start.tv_sec) * 1000000u + 
         end.tv_usec - start.tv_usec);
         
         }
         
         
         }
         ofstream out("TimesOldWay"); 
         out<<"Results for 100 time iterations:\n";
         for (int i=0; i<top; ++i) {
	 out<<"Price Grid\t"<<pow(2,i+3)<<"\tPrice\t"<<result[0][i]<<"\tclocktime\t"<<
	 T[0][i]/1e6<<"s\trealtime\t"<<real_T[0][i]/1e6<<"s\n";
	 out<<"LogPrice Grid\t"<<pow(2,i+3)<<"\tPrice\t"<<result[1][i]<<"\tclocktime\t"<<
	 T[1][i]/1e6<<"s\trealtime\t"<<real_T[1][i]/1e6<<"s\n";
         }
         
         */
        
        /*
         EuropeanOptionPrice<1> foo
         (OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 12, 250);
         EuropeanOptionLogPrice<1> foo2
         (OptionType::Put, model.get_pointer(), 0.0367, 1., 90., 12, 250);
         
         EuropeanOptionPrice<1> duffy
         (OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
         EuropeanOptionLogPrice<1> duffy2
         (OptionType::Call, model3.get_pointer(), 0.0367, 1., 90., 8, 100);
         
         AmericanOptionPrice<1> minnie
         (model.get_pointer(), 0.0367, 1., 90., 12, 100);
         AmericanOptionLogPrice<1> minnie2
         (model.get_pointer(), 0.0367, 1., 90., 12, 100);
         
         EuropeanOptionPrice<2> mickey
         (OptionType::Call, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 7, 100);
         
         EuropeanOptionLogPrice<2> mickey2
         (OptionType::Call, model1.get_pointer(), model2.get_pointer(),
         -0.2, 0.1, 1., 200., 7, 100);
         
         EuropeanOptionPrice<2> goofy
         (OptionType::Call, model5.get_pointer(), model4.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 25);
         
         EuropeanOptionLogPrice<2> goofy2
         (OptionType::Call, model5.get_pointer(), model4.get_pointer(),
         -0.2, 0.1, 1., 200., 6, 25);
         
         foo.run();
         foo2.run();
         duffy.run();
         duffy2.run();
         minnie.run();
         minnie2.run();
         mickey.run();
         mickey2.run();
         goofy.run();
         goofy2.run();
         
         cout<<foo.get_price()<<"\n";
         cout<<foo2.get_price()<<"\n";
         cout<<duffy.get_price()<<"\n";
         cout<<duffy2.get_price()<<"\n";
         cout<<minnie.get_price()<<"\n";
         cout<<minnie2.get_price()<<"\n";
         cout<<mickey.get_price()<<"\n";
         cout<<mickey2.get_price()<<"\n";
         cout<<goofy.get_price()<<"\n";
         cout<<goofy2.get_price()<<"\n";
         */
        /*
         auto x=OptionFactory::instance()->create(ExerciseType::EU,
         OptionType::Put,
         Transformation::Price,
         model.get_pointer(),
         0.0367, 1., 90., 12, 250);
         
         x->run();
         cout<<x->get_price()<<"\n";
         */
        cout<<"TARGET (Premia)\n"
        <<"PDE  1d Call  (10, 100): 9.62216.\n"
        <<"PDE  1d Put   (10, 100): 1.43609.\n"
        <<"PDE  1d PutAm (10, 100): 1.56720.\n"
        <<"PDE  2d Call  (7,  100): 21.3480.\n"
        <<"PIDE 1d Call  (8,  100): 12.3683.\n"
        <<"PIDE 2d Call  MC       : 24.8000.\n";
        
        return 0;
        
}