#include "Function.h"
#include "myRandom.h"

#include <cassert>
void Function::reduceDomains() {
	int new_num_values = Variable::getDomainSize(variables_);
	vector<Double> new_table(new_num_values);
	for (int i = 0; i < new_num_values; i++) {
		Variable::setAddress(variables_, i);
		int address = 0;
		int multiplier = 1;
		for (int j = 0; j < variables_.size(); j++) {
			address += (multiplier
					* variables_[j]->mapping[variables_[j]->addr_value()]);
			multiplier *= (int) variables_[j]->old_domain.size();
		}
		new_table[i] = table_[address];
	}
	table_ = new_table;
}
void Function::removeEvidence() {

	vector<Variable*> other_variables;
	for (int i = 0; i < variables_.size(); i++)
		if (variables_[i]->value() == INVALID_VALUE)
			other_variables.push_back(variables_[i]);

	int other_num_values = Variable::getDomainSize(other_variables);
	vector<Double> new_table(other_num_values);
	for (int j = 0; j < other_num_values; j++) {
		Variable::setAddress(other_variables, j);
		int entry = Variable::getAddress(variables_);
		new_table[j] = table()[entry];
	}
	variables_ = other_variables;
	table_ = new_table;
}
void Function::normalize() {
	Double norm_const = 0.0;
	for (int i = 0; i < table_.size(); i++) {
		norm_const += table_[i];
	}
	for (int i = 0; i < table_.size(); i++) {
		table_[i] /= norm_const;
	}
}

void Function::multiplyAndMarginalize(vector<Variable*>& marg_variables_, vector<Function*>& functions, Function& new_func,bool to_normalize)
{

	if (functions.empty()) return;
	vector<Variable*> variables;
	vector<Variable*> marg_variables;
	for(int i=0;i<functions.size();i++)
	{
		do_set_union(variables,functions[i]->variables(),variables,less_than_comparator_variable);
	}
	do_set_intersection(variables,marg_variables_,marg_variables,less_than_comparator_variable);
	//if (variables.empty()) return;
	new_func.variables()=marg_variables;
	new_func.table()=vector<Double>(Variable::getDomainSize(marg_variables),0);
	//cout<<"num-marg-vars = "<<marg_variables.size()<<" ";
	//if(variables.empty()){
		//new_func.table()=vector<Double>(1);
		//return;
	//}

	int num_variables=variables.size();
	int num_functions=functions.size();
	int num_marg_variables=marg_variables.size();

	// Compute gray index for all variables and functions
	vector<vector<pair<int,int> > > gray_index(num_variables);
	int old_temp_value[num_variables];
	for(int i=0;i<num_variables;i++)
	{
		old_temp_value[i]=variables[i]->temp_value;
		variables[i]->temp_value=i;
	}
	for(int i=0;i<num_functions;i++)
	{
		int multiplier=1;
		for(int j=0;j<functions[i]->variables().size();j++){
			gray_index[functions[i]->variables()[j]->temp_value].push_back(pair<int,int>(i,multiplier));
			multiplier*=functions[i]->variables()[j]->domain_size();
		}
	}

	//Initialize the data structure for gray code
	int a[num_variables];
	int f[num_variables+1];
	int o[num_variables+1];
	int m[num_variables];

	for(int i=0;i<num_variables;i++)
	{
		a[i]=0;
		f[i]=i;
		o[i]=1;
		m[i]=variables[i]->domain_size();
	}
	f[num_variables]=num_variables;
	int func_address[num_functions];
	int marg_address=0;
	Double mult(1.0);
	for(int i=0;i<num_functions;i++)
	{
		if(!functions[i]->table().empty())
			mult*=functions[i]->table()[0];
		func_address[i]=0;
	}
	int domain_size=Variable::getDomainSize(variables);
	int gray_marg_index[num_variables];

	for(int i=0;i<num_variables;i++)
		gray_marg_index[i]=0;
	int multiplier=1;
	for(int i=0;i<new_func.variables().size();i++)
	{
		gray_marg_index[new_func.variables()[i]->temp_value]=multiplier;
		multiplier*=new_func.variables()[i]->domain_size();
	}
	for(int i=0;i<domain_size;i++)
	{
		new_func.table()[marg_address]+=mult;
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
		for(int k=0;k<gray_index[j].size();k++)
		{
			int index=gray_index[j][k].first;
			int addr_multiplier=gray_index[j][k].second;
			func_address[index]-=addr_multiplier*old_aj;
			func_address[index]+=addr_multiplier*a[j];
		}
		mult=Double(1.0);
		for(int k=0;k<num_functions;k++)
		{
			mult*=functions[k]->table()[func_address[k]];
		}
		//End Hack
		if(gray_marg_index[j]>0)
		{
			marg_address-=gray_marg_index[j]*old_aj;
			marg_address+=gray_marg_index[j]*a[j];
		}
	}
	if (to_normalize){
		new_func.normalize();
	}
}
void Function::product(Function& function)
{

	if(function.table().empty() || function.variables().empty())
		return;
	vector<Variable*> new_variables;
	do_set_union(variables(),function.variables(),new_variables,less_than_comparator_variable);
	int num_values=Variable::getDomainSize(new_variables);
	if(new_variables.size()==variables().size())
	{
		for(int i=0;i<num_values;i++)
		{
			Variable::setAddress(variables(),i);
			int func_entry=Variable::getAddress(function.variables());
			table()[i]*=function.table()[func_entry];
		}
	}
	else
	{
		vector<Double> old_table;
		old_table=table_;
		table_=vector<Double> (num_values);
		for(int i=0;i<num_values;i++)
		{
			Variable::setAddress(new_variables,i);
			int entry=Variable::getAddress(variables());
			int func_entry=Variable::getAddress(function.variables());
			table()[i]=Double(function.table()[func_entry]*old_table[entry]);
		}
		variables_=new_variables;
	}
}
/*
void Function::multiplyAndMarginalize(vector<Variable*>& marg_variables_,
		vector<Function*>& functions, Function& out_function,
		bool to_normalize) {

	vector<Variable*> variables;
	vector<Variable*> marg_variables;
	int num_functions = functions.size();
	if (num_functions == 0)
		return;
	for (int i = 0; i < num_functions; i++) {
		do_set_union(variables, functions[i]->variables(), variables,
				less_than_comparator_variable);
	}
	int num_variables = variables.size();
	if (num_variables == 0)
		return;
	do_set_intersection(variables, marg_variables_, marg_variables,
			less_than_comparator_variable);
	int num_marg_variables = marg_variables.size();
	//if(num_marg_variables==0)
	//	return;

	// 6 arrays for graycoding using Knuth's algorithm
	vector<vector<pair<int, int> > > g(num_variables);
	int c[num_functions];
	int a[num_variables];
	int f[num_variables + 1];
	int o[num_variables];
	int m[num_variables];
	int t[num_variables];
	Double mult(1.0);
	int address = 0;

	// Init variables for graycoding
	for (int i = 0; i < num_variables; i++) {
		a[i] = 0;
		f[i] = i;
		o[i] = 1;
		t[i] = 0;
		m[i] = variables[i]->domain_size();
		variables[i]->temp_value = i;
	}
	for (int i = 0; i < num_functions; i++)
		c[i] = 0;
	f[num_variables] = num_variables;
	for (int i = 0; i < num_functions; i++) {
		int multiplier = 1;
		assert(functions[i]!=NULL);
		for (int j = 0; j < functions[i]->variables().size(); j++) {
			g[functions[i]->variables()[j]->temp_value].push_back(
					pair<int, int>(i, multiplier));
			multiplier *= functions[i]->variables()[j]->domain_size();
		}
		if (!functions[i]->table().empty()) {
			mult *= functions[i]->table()[0];
		}
	}
	int multiplier = 1;
	//cout<<"mult here\n";
	for (int i = 0; i < num_marg_variables; i++) {
		t[marg_variables[i]->temp_value] = multiplier;
		multiplier *= marg_variables[i]->domain_size();
	}
	//cout<<"mult initing log function\n";
	//Gray  code algorithm
	//Initialize LogFunction
	out_function.variables() = marg_variables;
	out_function.table() = vector<Double>(
			Variable::getDomainSize(marg_variables),0.0);

	//cout<<"Log function inited\n";
	while (1) {
		//cout<<address<<endl;
		// Step 1: Visit
		out_function.table()[address] += mult;
		// Step 2: Choose j
		int j = f[0];
		f[0] = 0;

		if (j == num_variables)
			break;
		int old_aj = a[j];
		a[j] = a[j] + o[j];
		if (a[j] == 0 || a[j] == (m[j] - 1)) {
			o[j] = -o[j];
			f[j] = f[j + 1];
			f[j + 1] = j + 1;
		}
		if (mult == 0.0){
			for (int i = 0; i < g[j].size(); i++) {
				int index = g[j][i].first;
				int multiplier = g[j][i].second;
				mult /= functions[index]->table()[c[index]];
				c[index] -= multiplier * old_aj;
				c[index] += multiplier * a[j];
				mult *= functions[index]->table()[c[index]];
			}
		}
		else {
			for (int i = 0; i < g[j].size(); i++) {
				int index = g[j][i].first;
				int multiplier = g[j][i].second;
				c[index] -= multiplier * old_aj;
				c[index] += multiplier * a[j];
			}
			mult = Double(1.0);
			for (int i = 0; i < num_functions; i++)
				mult *= functions[i]->table()[c[i]];
		}
		if (t[j] > 0) {
			address -= t[j] * old_aj;
			address += t[j] * a[j];
		}
	}
	if (to_normalize)
		out_function.normalize();
}
*/
