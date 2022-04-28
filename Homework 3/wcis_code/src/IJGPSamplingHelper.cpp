#include "IJGPSamplingHelper.h"

POS::POS(GM* gm, vector<int>& order,JG* jg)
{
  //cerr<<"Initing POS\n";
	vector<Variable*> all_variables;
	//cout<<"\t "<<gm->variables.size()<<endl;
	sampling_functions=vector<SF>(gm->variables.size()); 

	for(int i=0;i<gm->variables.size();i++)
	{
		int var=order[order.size()-1-i];
		all_variables.push_back(gm->variables[var]);
		sort(all_variables.begin(),all_variables.end(),less_than_comparator_variable);
		// Find the largest cluster in the join graph which has a non-zero intersection with all_variables
		int max_cluster_id=INVALID_VALUE;
		int max=INVALID_VALUE;
		for(int j=0;j<jg->nodes.size();j++)
		{
			bool found=false;
			for(int k=0;k<jg->nodes[j]->variables().size();k++)
			{
				if(jg->nodes[j]->variables()[k]->id()==var)
				{
					found=true;
					max_cluster_id=j;
					break;
				}
			}
			if(found)
			{
				//break;
				//cout<<"Variable found "<<var<<"\n";
				vector<Variable*> temp;
				do_set_intersection(all_variables,jg->nodes[j]->variables(),temp,less_than_comparator_variable);
				if((int)temp.size() > max)
				{
					max=(int)temp.size();
					max_cluster_id=j;
				}
			}
		}

		//max_cluster_id=INVALID_VALUE;
		if(max_cluster_id == INVALID_VALUE)
		{
			//jg->print(cout);
			//cout<<"invalid value for var = "<<var<<endl;
			vector<Variable*> cond_vars;
			CPT cpt;
			cpt.variables().push_back(gm->variables[var]);
			cpt.cond_variables()=vector<Variable*>();
			cpt.setMargVariable(gm->variables[var]);
			cpt.table()=vector<Double> (cpt.marg_variable()->domain_size());
			for(int j=0;j<cpt.table().size();j++)
			{
				cpt.table()[j]=Double((double)1.0/(double)cpt.marg_variable()->domain_size());
			}
			//cpt.epsilonCorrection(epsilon);
			sampling_functions[var]=SF(cpt);
			continue;
		}
		assert(max_cluster_id > INVALID_VALUE);
		//Form the function
		vector<Variable*> temp;
		do_set_intersection(all_variables,jg->nodes[max_cluster_id]->variables(),temp,less_than_comparator_variable);
		vector<Variable*> marg_vars;
		marg_vars.push_back(gm->variables[var]);
		do_set_difference(temp,marg_vars,temp,less_than_comparator_variable);
		
		CPT cpt;
		jg->nodes[max_cluster_id]->getCF(temp,gm->variables[var],cpt);
		Double epsilon((long double)0.01/(long double)cpt.marg_variable()->domain_size());
		//Double epsilon((long double)0.01);
		cpt.epsilonCorrection(epsilon);
		sampling_functions[var]=SF(cpt);
		
	}
}

void POS::getSample(const int& variable, int& value, Double& weight, myRandom& random)
{
	sampling_functions[variable].getSample(value,weight,random);	
}
Function& POS::getFunction(int variable) 
{
	return sampling_functions[variable];
}




IJGPSamplingHelper::IJGPSamplingHelper(GM* gm_, JG* jg_, int p_, vector<int>& order, SAMPLER_TYPE type)
:gm(gm_),
jg(jg_),
p(p_)
{
	//cerr<<"Propagating\n";
	//jg->propagate();
	//cerr<<"Propagation done\n";
	if(p<0)
	{
		p=0;
		//return;
	}
	else if(p > 100)
	{
		p=100;
	}
	sampler=new POS(gm_,order,jg_);
}
