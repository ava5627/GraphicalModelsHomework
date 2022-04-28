#include "BE.h"

void createFunction(Function& function, Function& out)
{
	out.variables().clear();
	out.table().clear();
	if(function.variables().empty() && function.table().empty())
	{
		out.table()=vector<Double>(1);
		out.table()[0]=Double(1.0);
		return;
	}
	//Remove all variables that have been assignend
	for(int i=0;i<function.variables().size();i++)
	{
		if(function.variables()[i]->value()==INVALID_VALUE)
			out.variables().push_back(function.variables()[i]);
	}
	if(out.variables().empty())
	{
		out.table()=vector<Double> (1);
		out.table()[0]=function.table()[Variable::getAddress(function.variables())];
		return;
	}
	if(out.variables().size()==function.variables().size())
	{
		out.table()=function.table();
	}
	else
	{
		int domain_size=Variable::getDomainSize(out.variables());
		out.table()=vector<Double>(domain_size);
		int num_variables=out.variables().size();
		int g[num_variables];
		int a[num_variables];
		int f[num_variables+1];
		int o[num_variables];
		int m[num_variables];

		int multiplier=1;
		for(int i=0;i<num_variables;i++)
		{
			a[i]=0;
			g[i]=multiplier;
			f[i]=i;
			o[i]=1;
			m[i]=out.variables()[i]->domain_size();
			multiplier*=m[i];
		}
		f[num_variables]=num_variables;
		int h[num_variables];
		int func_address=0;
		multiplier=1;
		int k=0;
		for(int i=0;i<function.variables().size();i++)
		{
			if(function.variables()[i]->value()!=INVALID_VALUE)
				func_address+=(multiplier*function.variables()[i]->value());
			else
			{
				assert(k<num_variables);
				h[k++]=multiplier;
			}
			multiplier*=function.variables()[i]->domain_size();
		}
		int address=0;

		while(1)
		{
			out.table()[address]=function.table()[func_address];
			int j=f[0];
			f[0]=0;

			if(j==num_variables) break;
			int old_aj=a[j];
			a[j]=a[j]+o[j];
			if(a[j]==0 || a[j]==(m[j]-1))
			{
				o[j]=-o[j];
				f[j]=f[j+1];
				f[j+1]=j+1;
			}
			address-=(g[j]*old_aj);
			address+=(g[j]*a[j]);
			func_address-=(h[j]*old_aj);
			func_address+=(h[j]*a[j]);
		}
	}
}

Double1 get_norm_const(Function& function)
{
	Double1 norm_const=0.0;
	for(int i=0;i<function.table().size();i++){
		norm_const+=function.table()[i];
	}
	for(int i=0;i<function.table().size();i++){
			function.table()[i]/=norm_const;
		}
	return norm_const;
}
BE::BE(std::vector<Variable*> &variables, std::vector<Function*> &functions, std::vector<int> &order)
{
	pe=Double(1.0);
	vector<vector<Function*> > buckets (order.size());

	vector<int> var_in_pos(order.size());
	for(int i=0;i<var_in_pos.size();i++)
		var_in_pos[order[i]]=i;

	// First put the functions in the proper buckets
    size_t width=0;
	for(int i=0;i<functions.size();i++)
	{
		int pos=order.size();
		//LogFunction* function=new LogFunction(*functions[i]);
		if(functions[i]->variables().empty())
		{
			pe*=functions[i]->table()[0];

			continue;
		}
		Function* function=new Function();
		createFunction(*functions[i],*function);
		for(int j=0;j<functions[i]->variables().size();j++)
		{
			if(var_in_pos[functions[i]->variables()[j]->id()] < pos)
				pos=var_in_pos[functions[i]->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		buckets[pos].push_back(function);
	}

	for(int i=0;i<buckets.size();i++)
	{
		if(buckets[i].empty())
			continue;

		vector<Variable*> bucket_variables;
		for(int j=0;j<buckets[i].size();j++)
		{
			do_set_union(bucket_variables,buckets[i][j]->variables(),bucket_variables,less_than_comparator_variable);
		}
        if (width<bucket_variables.size()) width=bucket_variables.size();
		//cout<<bucket_variables.size()<<" "<<flush;
		//cout<<"buck-vars.size ="<<bucket_variables.size()<<endl;
		//cerr<<"Processing bucket "<<i<<" out of "<<buckets.size()<<endl;


		// Compute variables required for marginalization
		//cerr<<bucket_variables.size()<<endl;
		vector<Variable*> bucket_variable;
		bucket_variable.push_back(variables[order[i]]);
		vector<Variable*> marg_variables;
		do_set_difference(bucket_variables,bucket_variable,marg_variables,less_than_comparator_variable);

		Function* function= new Function();
		Function::multiplyAndMarginalize(marg_variables,buckets[i],*function,false);

		if(function->variables().empty())
		{

			assert((int)function->table().size()==1);
			pe*=function->table()[0];
			delete(function);
			continue;
		}
		pe*=get_norm_const(*function);
		//Put the function in the appropriate bucket
		int pos=order.size();
		//function->print(cout);
		//assert(!function->log_table.empty());

		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		assert(pos > i);
		buckets[pos].push_back(function);
		for(int j=0;j<buckets[i].size();j++)
		{
			delete(buckets[i][j]);
		}
		buckets[i].clear();
	}
	for(int i=0;i<buckets.size();i++){
		for(int j=0;j<buckets[i].size();j++){
			if (buckets[i][j]!=NULL){
				delete(buckets[i][j]);
			}
		}
	}
	buckets.clear();
    cerr<<"Width along the given order = "<<width-1<<endl;
}

