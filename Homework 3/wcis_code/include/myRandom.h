#ifndef MY_RANDOM_H_
#define MY_RANDOM_H_


#include <random>
using namespace std;

struct myRandom {
    static default_random_engine randomEngine;
    static uniform_real_distribution<double> real_distribution;
    static uniform_int_distribution<int> int_distribution;

    // choose one of the random number generators:

    myRandom()=default;
    void setSeed(int seed_)
    {

    }
    double getDouble() {
        return real_distribution(randomEngine);
    }
    int getInt()
    {
        return int_distribution(randomEngine);
    }
    int getInt(int max_value) {
        return int_distribution(randomEngine)%max_value;
    }
};
#endif
