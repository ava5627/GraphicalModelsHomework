#include "BE.h"


BE::BE(std::vector<Variable*> &variables, std::vector<Function*> &functions, std::vector<int> &order)
{
	log_pe=LogDouble(1.0);
	vector<vector<LogFunction*> > buckets (order.size());

	vector<int> var_in_pos(order.size());
	for(int i=0;i<var_in_pos.size();i++)
		var_in_pos[order[i]]=i;

	// First put the functions in the proper buckets
	for(int i=0;i<functions.size();i++)
	{
		int pos=order.size();
		LogFunction* function=new LogFunction(*functions[i]);
		if(function->variables().empty())
		{
			//cerr<<"Deleting function\n";
			assert((int)function->log_table.size()==1);
			//cerr<<function->log_table[0].toDouble()<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}
		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		/*{
			assert((int)function->log_table.size()==1);
			cerr<<function->log_table[0].toDouble()<<" "<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}*/
		buckets[pos].push_back(function);
	}
	
	//cout<<"Now processing buckets\n";
	//Process buckets
	for(int i=0;i<buckets.size();i++)
	{
		if(buckets[i].empty())
			continue;

		vector<Variable*> bucket_variables;
		for(int j=0;j<buckets[i].size();j++)
		{
			do_set_union(bucket_variables,buckets[i][j]->variables(),bucket_variables,less_than_comparator_variable);
		}
		//cout<<bucket_variables.size()<<" "<<flush;
		//cout<<"buck-vars.size ="<<bucket_variables.size()<<endl;
		//cerr<<"Processing bucket "<<i<<" out of "<<buckets.size()<<endl;
		/*if((int)bucket_variables.size()==1)
		{
			Double temp;
			for(int k=0;k<bucket_variables[0]->domain_size();k++)
			{
				Double mult(1.0);
				for(int j=0;j<buckets[i].size();j++)
					mult*=buckets[i][j]->log_table[k].toDouble();
				temp+=mult;
			}
			cerr<<temp<<endl;
			log_pe+=LogDouble(temp);
			continue;
		}*/

		// Compute variables required for marginalization
		//cerr<<bucket_variables.size()<<endl;
		vector<Variable*> bucket_variable;
		bucket_variable.push_back(variables[order[i]]);
		vector<Variable*> marg_variables;
		do_set_difference(bucket_variables,bucket_variable,marg_variables,less_than_comparator_variable);

		LogFunction* function= new LogFunction();
		/*cout<<"Start mult\n";
		for(int j=0;j<buckets[i].size();j++)
		{
			cout<<"\t";
			buckets[i][j]->print(cout);
		}*/
		LogFunction::multiplyAndMarginalize(marg_variables,buckets[i],*function,false);
		//cout<<"End mult\n";

		if(function->variables().empty())
		{
			
			assert((int)function->log_table.size()==1);
			//cerr<<function->log_table[0].toDouble()<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}
		//Put the function in the appropriate bucket
		int pos=order.size();
		//function->print(cout);
		assert(!function->log_table.empty());

		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		/*if(pos==(int)order.size())
		{
			assert((int)function->log_table.size()==1);
			log_pe+=function->log_table[0];
			continue;
		}*/
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
}


BESample::BESample(std::vector<Variable*> &variables, std::vector<Function*> &functions, std::vector<int> &order,myRandom& random)
{
	log_pe=LogDouble(1.0);
	vector<vector<LogFunction*> > buckets (order.size());

	vector<int> var_in_pos(order.size());
	for(int i=0;i<var_in_pos.size();i++)
		var_in_pos[order[i]]=i;

	// First put the functions in the proper buckets
	for(int i=0;i<functions.size();i++)
	{
		int pos=order.size();
		LogFunction* function=new LogFunction(*functions[i]);
		if(function->variables().empty())
		{
			//cerr<<"Deleting function\n";
			assert((int)function->log_table.size()==1);
			//cerr<<function->log_table[0].toDouble()<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}
		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		/*{
			assert((int)function->log_table.size()==1);
			cerr<<function->log_table[0].toDouble()<<" "<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}*/
		buckets[pos].push_back(function);
	}
	
	//cout<<"Now processing buckets\n";
	//Process buckets
	for(int i=0;i<buckets.size();i++)
	{
		if(buckets[i].empty())
			continue;

		vector<Variable*> bucket_variables;
		for(int j=0;j<buckets[i].size();j++)
		{
			do_set_union(bucket_variables,buckets[i][j]->variables(),bucket_variables,less_than_comparator_variable);
		}
		//cout<<bucket_variables.size()<<" "<<flush;
		//cout<<"buck-vars.size ="<<bucket_variables.size()<<endl;
		//cerr<<"Processing bucket "<<i<<" out of "<<buckets.size()<<endl;
		/*if((int)bucket_variables.size()==1)
		{
			Double temp;
			for(int k=0;k<bucket_variables[0]->domain_size();k++)
			{
				Double mult(1.0);
				for(int j=0;j<buckets[i].size();j++)
					mult*=buckets[i][j]->log_table[k].toDouble();
				temp+=mult;
			}
			cerr<<temp<<endl;
			log_pe+=LogDouble(temp);
			continue;
		}*/

		// Compute variables required for marginalization
		//cerr<<bucket_variables.size()<<endl;
		vector<Variable*> bucket_variable;
		bucket_variable.push_back(variables[order[i]]);
		vector<Variable*> marg_variables;
		do_set_difference(bucket_variables,bucket_variable,marg_variables,less_than_comparator_variable);

		LogFunction* function= new LogFunction();
		/*cout<<"Start mult\n";
		for(int j=0;j<buckets[i].size();j++)
		{
			cout<<"\t";
			buckets[i][j]->print(cout);
		}*/
		LogFunction::multiplyAndMarginalize(marg_variables,buckets[i],*function,false);
		//cout<<"End mult\n";

		if(function->variables().empty())
		{
			
			assert((int)function->log_table.size()==1);
			//cerr<<function->log_table[0].toDouble()<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}
		//Put the function in the appropriate bucket
		int pos=order.size();
		//function->print(cout);
		assert(!function->log_table.empty());

		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		/*if(pos==(int)order.size())
		{
			assert((int)function->log_table.size()==1);
			log_pe+=function->log_table[0];
			continue;
		}*/
		assert(pos > i);
		buckets[pos].push_back(function);
		/*
		for(int j=0;j<buckets[i].size();j++)
		{
			delete(buckets[i][j]);
		}
		buckets[i].clear();
		*/
	}

	/* Generate a sample from the buckets */
	for(int i=buckets.size()-1;i>-1;i--){
		int curr_var=order[i];
		if(variables[curr_var]->value()!=INVALID_VALUE){
			continue;
		}
		vector<Double> marginal(variables[curr_var]->domain_size());
		for (int j=0;j<marginal.size();j++)
			marginal[j]=Double(1.0);
		for(int j=0;j<buckets[i].size();j++)
		{
			for(int k=0;k<buckets[i][j]->variables().size();k++)
			{
				if (buckets[i][j]->variables()[k]->id()==curr_var)
					continue;
				assert(buckets[i][j]->variables()[k]->value()!=INVALID_VALUE);
			}
			for(int k=0;k<marginal.size();k++)
			{
				int entry;
				variables[curr_var]->addr_value()=k;
				entry=Variable::getAddress(buckets[i][j]->variables());
				marginal[k]*=buckets[i][j]->log_table[entry].toDouble();
			}
		}
		Double norm_const;
		for(int j=0;j<marginal.size();j++){
			norm_const+=marginal[j];
		}
		double rand_num = random.getDouble();
		Double cdf;
		for(int j=0;j<marginal.size();j++){
			marginal[j]/=norm_const;
			cdf+=marginal[j];
			if (rand_num <= cdf.value()){
				variables[curr_var]->value()=j;
				break;
			}
		}
		assert(variables[curr_var]->value()!=INVALID_VALUE);
	}
	/* Delete all the buckets */
	for(int i=0;i<buckets.size();i++){
		for(int j=0;j<buckets[i].size();j++){
			if (buckets[i][j]!=NULL){
				delete(buckets[i][j]);
			}
		}
	}
	buckets.clear();
}






BucketProp::BucketProp(std::vector<Variable*> &variables, std::vector<Function*> &functions, std::vector<int> &order)
{
	log_pe=LogDouble(1.0);
	vector<vector<LogFunction*> > buckets (order.size());

	vector<int> var_in_pos(order.size());
	for(int i=0;i<var_in_pos.size();i++)
		var_in_pos[order[i]]=i;

	// First put the functions in the proper buckets
	for(int i=0;i<functions.size();i++)
	{
		int pos=order.size();
		LogFunction* function=new LogFunction(*functions[i]);
		if(function->variables().empty())
		{
			//cerr<<"Deleting function\n";
			assert((int)function->log_table.size()==1);
			//cerr<<function->log_table[0].toDouble()<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}
		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		/*{
			assert((int)function->log_table.size()==1);
			cerr<<function->log_table[0].toDouble()<<" "<<endl;
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}*/
		buckets[pos].push_back(function);
	}

	//cout<<"Now processing buckets\n";
	//Process buckets
	vector<pair<int,int> > messages;
	vector<int> to_not_include;
	vector<vector<Variable*> > all_bucket_vars (buckets.size());
	for(int i=0;i<buckets.size();i++)
	{
		if(buckets[i].empty())
			continue;

		vector<Variable*> bucket_variables;
		for(int j=0;j<buckets[i].size();j++)
		{
			do_set_union(bucket_variables,buckets[i][j]->variables(),bucket_variables,less_than_comparator_variable);
		}
		// Compute variables required for marginalization
		vector<Variable*> bucket_variable;
		bucket_variable.push_back(variables[order[i]]);
		vector<Variable*> marg_variables;
		do_set_difference(bucket_variables,bucket_variable,marg_variables,less_than_comparator_variable);
		all_bucket_vars[i]=bucket_variables;
		LogFunction* function= new LogFunction();
		LogFunction::multiplyAndMarginalize(marg_variables,buckets[i],*function,false);
		if(function->variables().empty())
		{
			assert((int)function->log_table.size()==1);
			log_pe+=function->log_table[0];
			delete(function);
			continue;
		}
		//Put the function in the appropriate bucket
		int pos=order.size();
		assert(!function->log_table.empty());
		for(int j=0;j<function->variables().size();j++)
		{
			if(var_in_pos[function->variables()[j]->id()] < pos)
				pos=var_in_pos[function->variables()[j]->id()];
		}
		assert(pos!=(int)order.size());
		assert(pos > i);
		messages.push_back(pair<int,int> (i,pos));
		to_not_include.push_back(buckets[pos].size());
		buckets[pos].push_back(function);
	}
	for(int i=messages.size()-1;i>-1;i--){
		int from_node=messages[i].second;
		int to_node=messages[i].first;
		vector<Variable*> marg_variables;
		do_set_intersection(all_bucket_vars[from_node],all_bucket_vars[to_node],marg_variables,less_than_comparator_variable);
		LogFunction* function= new LogFunction();
		// Create a vector of all functions except the to_not_include function
		vector<LogFunction*> curr_bucket_functions;
		bool found=false;
		for(int j=0;j<buckets[from_node].size();j++){
			if(j==to_not_include[i]){
				found=true;
			}
			curr_bucket_functions.push_back(buckets[from_node][j]);
		}
		LogFunction::multiplyAndMarginalize(marg_variables,curr_bucket_functions,*function,true);
		if (function->variables().empty()) {
			assert((int)function->log_table.size()==1);
			delete (function);
			continue;
		}
		buckets[to_node].push_back(function);
	}
	marginals=vector<Function> (order.size());
	for(int i=0;i<order.size();i++){
		int var=order[i];
		vector<Variable*> marg_variable;
		marg_variable.push_back(variables[var]);
		if (variables[order[i]]->value()!=INVALID_VALUE){
			marginals[var].variables()=marg_variable;
			marginals[var].table()=vector<Double> (variables[var]->domain_size());
			marginals[var].table()[variables[var]->value()]=Double(1.0);
		}
		else{
			LogFunction::multiplyAndMarginalize(marg_variable,buckets[i],marginals[var],true);
			if (marginals[var].table().empty()){
				marginals[var].variables()=marg_variable;
				marginals[var].table()=vector<Double> (variables[var]->domain_size());
				for(int j=0;j<variables[var]->domain_size();j++){
					marginals[var].table()[j]=Double((long double)1.0/(long double)variables[var]->domain_size());
				}
			}
		}
	}
	for(int i=0;i<buckets.size();i++){
		for(int j=0;j<buckets[i].size();j++){
			if (buckets[i][j]!=NULL){
				delete(buckets[i][j]);
			}
		}
	}
	buckets.clear();
}
