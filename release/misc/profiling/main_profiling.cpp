#include <iostream>
#include "Levy.hpp"

int main()
{

        using namespace dealii;
        using namespace std;

        KouModel model(95, 0.120381, 0.20761, 0.330966, 9.65997, 3.13868);

        omp_set_num_threads(1);

        auto foo=Factory::instance()->create(ExerciseType::EU,
                                             OptionType::Call,
                                             Transformation::LogPrice,
                                             model.get_pointer(),
                                             0.0367, 1., 90., 8, 100);

        foo->set_verbose(false);
        foo->set_integral_adaptivity_params(false, 16);

        foo->run();

        return 0;

}
