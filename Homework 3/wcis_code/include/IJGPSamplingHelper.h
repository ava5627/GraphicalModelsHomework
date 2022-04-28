#ifndef IJGPSamplingHelper_H_
#define IJGPSamplingHelper_H_

#include "GM.h"
#include "JG.h"
#include "SF.h"
#include "CPT.h"
typedef enum { POSITIVE_SAMPLER, ZERO_SAMPLER} SAMPLER_TYPE;
// Abstract class Ordered Sampler
class OS
{
public:
	OS() { }
	virtual ~OS(){ }
	virtual void getSample(const int& variable, int& value, Double& weight,myRandom& random){ }
	virtual Function& getFunction(int variable)=0;
};
// Positive Ordered sampler
class POS: public OS
{
private:
	vector<SF> sampling_functions;
public:
	POS(GM* gm_, vector<int>& order,JG* jg_);
	~POS(){ }
	void getSample(const int& variable, int& value, Double& weight,myRandom& random);
	Function& getFunction(int variable);
};
// Positive Ordered Sampler with parameter p
class POSP: public OS
{
private:
	int p;
	vector<JGNode*> var_nodes;
public:
	POSP(GM* gm_, vector<int>& order,JG* jg_,int p_);
	~POSP(){ }
	void getSample(const int& variable, int& value, Double& weight,myRandom& random);
	Function& getFunction(int variable);

};


class IJGPSamplingHelper
{
protected:
	GM* gm;
	JG* jg;
	int p;
	OS* sampler;
	// Pointer to the join graph nodes for each variable
public:
	IJGPSamplingHelper(){}
	IJGPSamplingHelper(GM* gm_, JG* jg_, int p_, vector<int>& order, SAMPLER_TYPE type=ZERO_SAMPLER);
	~IJGPSamplingHelper() {delete(sampler);}
	Function& getFunction(int variable){ return sampler->getFunction(variable);}
	void getSample(const int& variable, int& value, Double& weight,myRandom& random)
	{
		sampler->getSample(variable,value,weight,random);
	}
};
#endif
