#ifndef BE_H_
#define BE_H_

#include <vector>
#include "GM.h"
struct BE
{
	Double1 pe;
	BE(vector<Variable*>& variables, vector<Function*>& functions, vector<int>& order);
	
};
#endif
