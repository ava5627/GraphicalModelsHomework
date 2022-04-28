//
// Created by Vibhav Gogate on 3/21/22.
//

#include "myRandom.h"
#include <cstdlib>
default_random_engine myRandom::randomEngine=default_random_engine{};
uniform_real_distribution<double> myRandom::real_distribution=uniform_real_distribution<double> (0.0, 1.0);
uniform_int_distribution<int> myRandom::int_distribution=uniform_int_distribution<int> (0, RAND_MAX);