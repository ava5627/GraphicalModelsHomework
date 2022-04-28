#ifndef MY_RANDOM_H_
#define MY_RANDOM_H_

//#include "randomc.h"
#include <cstdlib>
using namespace std;

struct myRandom {
	int seed;
	// choose one of the random number generators:

	myRandom() {
		//seed=time(NULL);
		//seed=1038383883L;
		//srand(seed);
	}
	void setSeed(int seed_)
	{
		//seed=time(NULL);
		//seed=1038383883L;
		//srand(seed);
	}
	double getDouble() {
		return drand48();
	}
	int getInt()
	{
		return lrand48();
	}
	int getInt(int max_value) {
		return lrand48()%max_value;
	}
};
/*
struct myRandom {
	int seed;
	CRandomMersenne RanGen;
	// choose one of the random number generators:

	myRandom() {
		//seed=time(NULL);
		seed=1000;
		RanGen = CRandomMersenne(seed);
	}
	void setSeed(int seed_)
	{
		//seed=time(NULL);
		seed=1000;
		RanGen = CRandomMersenne(seed);
	}
	double getDouble() {
		return RanGen.Random();
	}
	int getInt()
	{
		return RanGen.BRandom();
	}
	int getInt(int max_value) {
		RanGen.IRandomX(0,max_value-1);
	}
};
*/
#endif
