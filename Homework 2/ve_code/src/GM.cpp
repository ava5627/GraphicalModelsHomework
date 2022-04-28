#include "GM.h"
#include "myRandom.h"
#include "LogDouble.h"
#include "Graph.h"
struct less_than_comparator_function {
	bool operator()(const Function* a, const Function* b) const {
		return (a->id() < b->id());
	}
};

void GM::readUAI08(const char* infilename) {
	ifstream infile(infilename);
    if (!infile.good()){
        cerr<<"Network file "<<infilename<<" not found\n";
        exit(1);
    }
    else{
        cerr<<"Using network "<<infilename<<endl;
    }
	int num_variables;
	string tmp_string;
	infile >> tmp_string;
	if (tmp_string.compare("BAYES") == 0) {
		type = BAYES;
	} else if (tmp_string.compare("MARKOV") == 0) {
		type = MARKOV;
	}

	if (type == BAYES) {
		//cerr << "Reading bayesnet\n";
		infile >> num_variables;
		// Read domains
		variables = vector<Variable*>(num_variables);
		for (int i = 0; i < num_variables; i++) {
			int domain_size;
			infile >> domain_size;
			vector<int> domain(domain_size);
			for (int j = 0; j < domain_size; j++)
				domain[j] = j;
			variables[i] = new Variable(i, domain);
			variables[i]->orig_id = i;
		}
		copy_of_variables = variables;
		int num_functions;
		infile >> num_functions;
		vector<vector<Variable*> > parents(num_variables);
		vector<int> func_order(num_functions);
		for (int i = 0; i < num_functions; i++) {
			// Read parents of variables
			int num_parents;
			infile >> num_parents;
			num_parents--;
			vector<Variable*> curr_parents(num_parents);
			for (int j = 0; j < num_parents; j++) {
				int temp;
				infile >> temp;
				curr_parents[j] = variables[temp];
			}
			int var;
			infile >> var;
			func_order[i] = var;
			parents[var] = curr_parents;
		}
		functions = vector<Function*>(num_functions);
		for (int i = 0; i < num_functions; i++) {

			int var = func_order[i];
			int num_probabilities;
			infile >> num_probabilities;

			functions[var] = new CPT();
			CPT* cpt = dynamic_cast<CPT*>(functions[var]);
			cpt->id() = var;
			cpt->setMargVariable(variables[var]);
			cpt->cond_variables() = parents[var];
			cpt->variables() = cpt->cond_variables();
			cpt->variables().push_back(cpt->marg_variable());
			sort(cpt->variables().begin(), cpt->variables().end(),
					less_than_comparator_variable);
			int cond_num_values = Variable::getDomainSize(parents[var]);
			assert(
					num_probabilities == (cond_num_values*cpt->marg_variable()->domain_size()));
			cpt->table() = vector<Double>(num_probabilities);
			for (int j = 0; j < cond_num_values; j++) {
				Variable::setAddressVIB(cpt->cond_variables(), j);
				for (int k = 0; k < cpt->marg_variable()->domain_size(); k++) {
					cpt->marg_variable()->addr_value() = k;
					double value;
					infile >> value;

					int entry = Variable::getAddress(cpt->variables());
					cpt->table()[entry] = value;
					if (value==0.0){
						mode = DET;
					}
				}
			}

			sort(cpt->cond_variables().begin(), cpt->cond_variables().end(),
					less_than_comparator_variable);

		}

	} else if (type == MARKOV) {
		//cerr << "Reading Markov network\n";
		infile >> num_variables;
		// Read domains
		variables = vector<Variable*>(num_variables);
		for (int i = 0; i < num_variables; i++) {
			int domain_size;
			infile >> domain_size;
			vector<int> domain(domain_size);
			for (int j = 0; j < domain_size; j++)
				domain[j] = j;
			variables[i] = new Variable(i, domain);
			variables[i]->orig_id = i;
		}
		copy_of_variables = variables;
		int num_functions;
		infile >> num_functions;
		vector<vector<Variable*> > scope(num_functions);
		for (int i = 0; i < num_functions; i++) {
			// Read parents of variables
			int num_vars_in_func;
			infile >> num_vars_in_func;
			scope[i] = vector<Variable*>(num_vars_in_func);
			for (int j = 0; j < num_vars_in_func; j++) {
				int temp;
				infile >> temp;
				scope[i][j] = variables[temp];
			}
		}
		functions = vector<Function*>(num_functions);
		for (int i = 0; i < num_functions; i++) {
			int var = i;
			int num_probabilities;
			infile >> num_probabilities;

			functions[var] = new Function(var, scope[i]);
			Function* cpt = functions[var];

			/*
			 cout<<num_probabilities<<": "<<variables[var]->id()<<" |";
			 for(int j=0;j<parents[var].size();j++)
			 cout<<parents[var][j]->id()<<",";
			 cout<<endl;
			 */

			sort(cpt->variables().begin(), cpt->variables().end(),
					less_than_comparator_variable);

			//int num_values=Variable::getDomainSize(sorted_func_variables);
			////int marg_num_values=Variable::getNumValues(marg_variables);
			//functions[i]->variables()=sorted_func_variables;
			int num_values = Variable::getDomainSize(scope[var]);
			assert(num_probabilities == num_values);
			cpt->table() = vector<Double>(num_probabilities);
			for (int j = 0; j < num_probabilities; j++) {
				Variable::setAddressVIB(scope[var], j);
				double value;
				infile >> value;
				int entry = Variable::getAddress(cpt->variables());
				cpt->table()[entry] = value;
				if (value==0.0){
					mode = DET;
				}
			}
		}
	}
}

void GM::setEvidenceBeliefsUAI08(vector<int>& evidence) {
	mult_factor = 1.0;
	vector<Function*> new_functions;
	for (auto & function : functions) {
		function->removeEvidence();
		if (function->variables().empty()) {
			mult_factor *= function->table()[0];
			delete function;
			continue;
		}
		new_functions.push_back(function);
	}
	functions = new_functions;
	vector<Variable*> new_variables;
	int count = 0;
	for (auto & variable : variables) {
		if (variable->value() == INVALID_VALUE) {
			new_variables.push_back(variable);
			variable->id() = count;
			count++;
		} else {
			vector<bool> new_domains(variable->domain_size());
			for (int j = 0; j < variable->domain_size(); j++) {
				if (variable->value() == j) {
					new_domains[j] = true;
				} else {
					new_domains[j] = false;
				}
			}
			variable->updateDomain(new_domains);
			variable->value() = 0;
			variable->id() = INVALID_VALUE;
		}
	}
	variables = new_variables;
}


void GM::getLexOrdering(vector<int>& order, vector<set<int> >& clusters,
		double& estimate) {
	estimate = 0.0;
	order = vector<int>(variables.size());
	clusters = vector<set<int> >(variables.size());
	vector<vector<bool> > adj_matrix(variables.size());
	for (int i = 0; i < variables.size(); i++) {
		adj_matrix[i] = vector<bool>(variables.size());
	}
	vector<set<int> > graph(variables.size());

	for (auto & function : functions) {
		for (int j = 0; j < function->variables().size(); j++) {
			for (int k = j + 1; k < function->variables().size(); k++) {
				int a = function->variables()[j]->id();
				int b = function->variables()[k]->id();
				graph[a].insert(b);
				graph[b].insert(a);
				adj_matrix[a][b] = true;
				adj_matrix[b][a] = true;
			}
		}
	}
	cout << "lex inited\n";
	//cerr<<"minfill: Inited\n";
	for (int i = 0; i < variables.size(); i++) {
		int min_id = i;
		order[i] = min_id;

		// Now form the cluster 
		clusters[i] = graph[min_id];
		clusters[i].insert(min_id);

		// Trinagulate min id and remove it from the graph
		for (auto a = graph[min_id].begin();
				a != graph[min_id].end(); a++) {
			auto b = a;
			b++;
			for (; b != graph[min_id].end(); b++) {
				if (!adj_matrix[*a][*b]) {
					adj_matrix[*a][*b] = true;
					adj_matrix[*b][*a] = true;
					graph[*a].insert(*b);
					graph[*b].insert(*a);
				}
			}
		}
		for (auto a = graph[min_id].begin();
				a != graph[min_id].end(); a++) {
			graph[*a].erase(min_id);
			adj_matrix[*a][min_id] = false;
			adj_matrix[min_id][*a] = false;
		}
		graph[min_id].clear();
	}
	cout << "Lex done\n";
	// compute the estimate
	int max_cluster_size = 0;
	for (auto & cluster : clusters) {
		if ((int) cluster.size() > max_cluster_size)
			max_cluster_size = (int) cluster.size();
		double curr_estimate = 1.0;
		for (int j : cluster) {
			curr_estimate *= (double) variables[j]->domain_size();
		}
		estimate += curr_estimate;
	}
	cerr << "Max cluster size =" << max_cluster_size << endl;
	cerr << "Estimate  = " << estimate << endl;
}

void GM::printMarginalsUAI10(std::vector<vector<Double1> > &marginals_,
		ostream& out) {
	vector<vector<Double1> > marginals = marginals_;
	//Normalize Marginals
	for (auto & marginal : marginals) {
		Double1 norm_const=0.0;
		for (long double j : marginal)
			norm_const += j;
		for (long double & j : marginal)
			j /= norm_const;
	}
	out << copy_of_variables.size() << " ";
	for (auto & copy_of_variable : copy_of_variables) {
		out << copy_of_variable->old_domain.size() << " ";
		if (copy_of_variable->value() == INVALID_VALUE) {
			int curr_id = copy_of_variable->id();
			vector<Double1> var_marginals(
					copy_of_variable->old_domain.size());
			int old_num = 0;
			for (int j = 0; j < marginals[curr_id].size(); j++) {
				int new2old = copy_of_variable->mapping[j];
				var_marginals[new2old] = marginals[curr_id][j];
			}
			for (long double var_marginal : var_marginals) {
				out << var_marginal << " ";
			}
		} else {
			if (copy_of_variable->domain_size() != 1) {
				cerr << copy_of_variable->domain_size() << endl;
			}assert((int)copy_of_variable->domain_size()==1);
			assert((int)copy_of_variable->value()==0);
			int val = copy_of_variable->mapping[0];
			for (int j = 0; j < copy_of_variable->old_domain.size(); j++) {
				if (j == val) {
					out << "1 ";
				} else {
					out << "0 ";
				}
			}
		}
	}
	out << endl;
}


void GM::getMinFillOrdering(vector<int>& order,
		vector<set<int> >& clusters, double& estimate, int& max_cluster_size) {
	estimate = 0.0;
	max_cluster_size = 0;
	order = vector<int>(variables.size());
	clusters = vector<set<int> >(variables.size());
	vector<vector<bool> > adj_matrix(variables.size());

	// Create the interaction graph of the functions in this graphical model - i.e.
	// create a graph structure such that an edge is drawn between variables in the
	// model that appear in the same function
	for (int i = 0; i < variables.size(); i++) {
		adj_matrix[i] = vector<bool>(variables.size());
	}
	vector<set<int> > graph(variables.size());
	vector<bool> processed(variables.size());
	for (int i = 0; i < functions.size(); i++) {
		for (int j = 0; j < functions[i]->variables().size(); j++) {
			for (int k = j + 1; k < functions[i]->variables().size(); k++) {
				int a = functions[i]->variables()[j]->id();
				int b = functions[i]->variables()[k]->id();
				graph[a].insert(b);
				graph[b].insert(a);
				adj_matrix[a][b] = true;
				adj_matrix[b][a] = true;
			}
		}
	}
	list<int> zero_list;
	//cerr<<"minfill: Inited\n";

	// For i = 1 to number of variables in the model
	// 1) Identify the variables that if deleted would add the fewest number of edges to the
	//    interaction graph
	// 2) Choose a variable, pi(i), from among this set
	// 3) Add an edge between every pair of non-adjacent neighbors of pi(i)
	// 4) Delete pi(i) from the interaction graph
	for (int i = 0; i < variables.size(); i++) {
		// Find variables with the minimum number of edges added
		double min = DBL_MAX;
		int min_id = -1;
		bool first = true;

		// Flag indicating whether the variable to be removed is from the
		// zero list - i.e. adds no edges to interaction graph when deleted
		bool fromZeroList = false;

		// Vector to keep track of the ID of each minimum fill variable
		vector<int> minFillIDs;

		// If there are no variables that, when deleted, add no edges...
		if (zero_list.empty()) {

			// For each unprocessed (non-deleted) variable
			for (int j = 0; j < variables.size(); j++) {
				if (processed[j])
					continue;
				double curr_min = 0.0;
				for (auto a = graph[j].begin();
						a != graph[j].end(); a++) {
					auto b = a;
					b++;
					for (; b != graph[j].end(); b++) {
						if (!adj_matrix[*a][*b]) {
							curr_min += (variables[*a]->domain_size()
									* variables[*b]->domain_size());
							if (curr_min > min)
								break;
						}
					}
					if (curr_min > min)
						break;
				}

				// Store the first non-deleted variable as a potential minimum
				if (first) {
					minFillIDs.push_back(j);
					min = curr_min;
					first = false;
				} else {
					// If this is a new minimum...
					if (min > curr_min) {
						min = curr_min;
						minFillIDs.clear();
						minFillIDs.push_back(j);
					}
					// Otherwise, if the number of edges removed is also a minimum, but
					// the minimum is zero
					else if (curr_min < DBL_MIN) {
						zero_list.push_back(j);
					}
					// Else if this is another potential min_fill
					else if (min == curr_min) {
						minFillIDs.push_back(j);
					}
				}
			}
		}
		// Else...delete variables from graph that don't add any edges
		else {
			min_id = zero_list.front();
			zero_list.pop_front();
			fromZeroList = true;
		}

		// If not from zero_list, choose one of the variables at random
		// from the set of min fill variables
		if (!fromZeroList) {
			int indexInVector = rand() % (int) minFillIDs.size();
			min_id = minFillIDs[indexInVector];
		}

		//cout<<"order["<<i<<"]= "<<min_id<<" "<<flush;
		assert(min_id!=-1);
		order[i] = min_id;
		// Now form the cluster
		clusters[i] = graph[min_id];
		clusters[i].insert(min_id);

		// Trinagulate min id and remove it from the graph
		for (auto a = graph[min_id].begin();
				a != graph[min_id].end(); a++) {
			auto b = a;
			b++;
			for (; b != graph[min_id].end(); b++) {
				if (!adj_matrix[*a][*b]) {
					adj_matrix[*a][*b] = true;
					adj_matrix[*b][*a] = true;
					graph[*a].insert(*b);
					graph[*b].insert(*a);
				}
			}
		}
		for (auto a = graph[min_id].begin();
				a != graph[min_id].end(); a++) {
			graph[*a].erase(min_id);
			adj_matrix[*a][min_id] = false;
			adj_matrix[min_id][*a] = false;
		}
		graph[min_id].clear();
		processed[min_id] = true;
	}

	// compute the estimate
	for (auto & cluster : clusters) {
		if ((int) cluster.size() > max_cluster_size)
			max_cluster_size = (int) cluster.size();
		double curr_estimate = 1.0;
		for (int j : cluster) {
			curr_estimate *= (double) variables[j]->domain_size();
		}
		estimate += curr_estimate;
	}
}


void GM::rearrangeOrdering_randomized(std::vector<int> &order,
		std::vector<set<int> > &clusters, std::vector<int> &new_order,
		double& log_limit) {
	new_order.clear();
	vector<vector<int> > var2clusters(variables.size());

	vector<LogDouble> var2estimate(variables.size());
	LogDouble limit(exp((long double) log_limit));
	LogDouble estimate;
	vector<bool> processed(variables.size());
	vector<LogDouble> cluster_estimate(clusters.size());

	for (int i = 0; i < clusters.size(); i++) {
		processed[i] = false;
		cluster_estimate[i] = Double(1.0);
		for (set<int>::iterator j = clusters[i].begin(); j != clusters[i].end();
				j++) {
			var2clusters[*j].push_back(i);
			cluster_estimate[i] *= LogDouble(
					(long double) variables[*j]->domain_size());
		}
		for (set<int>::iterator j = clusters[i].begin(); j != clusters[i].end();
				j++) {
			var2estimate[*j] += cluster_estimate[i];
		}
		estimate += cluster_estimate[i];
	}
	int count = 0;
	myRandom random;
	while (estimate > limit) {
		if (new_order.size() == order.size())
			break;
		LogDouble max;
		int min_id = -1;
		bool first = true;
		// Vector to keep track of the ID of each minimum
		vector<int> minIDs;
		for (int i = 0; i < variables.size(); i++) {
			if (processed[i])
				continue;
			if (first) {
				first = false;
				max = var2estimate[i];
				minIDs.push_back(i);
			} else {
				if (max < var2estimate[i]) {
					max = var2estimate[i];
					minIDs.clear();
					minIDs.push_back(i);
				} else if (var2estimate[i] < max) {
					continue;
				} else {
					minIDs.push_back(i);
				}
			}
		}
		int indexInVector = random.getInt((int) minIDs.size());
		min_id = minIDs[indexInVector];

		assert(min_id!=-1);
		processed[min_id] = true;
		new_order.push_back(min_id);

		//Update cluster2estimate, var2estimate and estimate
		for (int i = 0; i < var2clusters[min_id].size(); i++) {
			int cluster_id = var2clusters[min_id][i];
			estimate -= cluster_estimate[cluster_id];
			for (set<int>::iterator j = clusters[cluster_id].begin();
					j != clusters[cluster_id].end(); j++) {
				if (!processed[*j]) {
					var2estimate[*j] -= cluster_estimate[cluster_id];
				}
			}
			cluster_estimate[cluster_id] /= LogDouble(
					(long double) variables[min_id]->domain_size());
			estimate += cluster_estimate[cluster_id];
			for (set<int>::iterator j = clusters[cluster_id].begin();
					j != clusters[cluster_id].end(); j++) {
				if (!processed[*j]) {
					var2estimate[*j] += cluster_estimate[cluster_id];
				}
			}
		}
	}
}

void GM::printMarginalsUAI10(std::vector<Function> &marginals_, ostream& out)
{
	vector<vector<Double> > marginals(marginals_.size());
	for (int i = 0; i < marginals_.size(); i++) {
		marginals[i] = vector<Double> (marginals_[i].table().size());
		Double norm_const=0.0;
		for (int j = 0; j < marginals_[i].table().size(); j++) {
			norm_const += marginals_[i].table()[j];
			marginals[i][j] = marginals_[i].table()[j];
		}
		for (int j = 0; j < marginals[i].size(); j++)
			marginals[i][j] /= norm_const;
	}
	out<<copy_of_variables.size()<<" ";
	for(int i=0;i<copy_of_variables.size();i++)
	{
		out<<copy_of_variables[i]->old_domain.size()<<" ";
		if(copy_of_variables[i]->value()==INVALID_VALUE)
		{
			int curr_id=copy_of_variables[i]->id();
			vector<Double> var_marginals(copy_of_variables[i]->old_domain.size());
			int old_num=0;
			for(int j=0;j<marginals[curr_id].size();j++)
			{
				int new2old=copy_of_variables[i]->mapping[j];
				var_marginals[new2old]=marginals[curr_id][j];
			}
			for(int j=0;j<var_marginals.size();j++){
				out<<var_marginals[j]<<" ";
			}
		}
		else
		{
			assert((int)copy_of_variables[i]->domain_size()==1);
			assert((int)copy_of_variables[i]->value()==0);
			int val=copy_of_variables[i]->mapping[0];
			for(int j=0;j<copy_of_variables[i]->old_domain.size();j++){
				if (j==val){
					out<<"1 ";
				}
				else{
					out<<"0 ";
				}
			}
		}
	}
	out<<endl;
}

void GM::funcReduce()
{
	cerr<<"num functions before processing = "<<functions.size()<<endl;
	vector<Function*> new_functions;
	vector<bool> is_deleted(functions.size());
	mode=POSITIVE;

	for (int i=0;i<functions.size();i++){
		for(int j=0;j<functions[i]->table().size();j++){
			if (functions[i]->table()[j]==0.0){
				mode=DET;
			}
			if (mode==DET) break;
		}
		if (mode==DET) break;
	}
	for (int i=0;i<functions.size();i++){
		functions[i]->normalize();
	}
	for (int i = 0; i < functions.size(); i++) {
		Double some_value(1);
		bool same = true;
		for (int j = 0; j < functions[i]->table().size(); j++) {
			if (j == 0) {
				some_value = functions[i]->table()[j];
			} else if (is_equal(some_value, functions[i]->table()[j])) {
				continue;
			} else {
				same = false;
				break;
			}
		}
		if (same && !functions[i]->table().empty()) {
			this->mult_factor *= some_value
					* Double(functions[i]->table().size());
			is_deleted[i] = true;
		}
	}
	for (int i=0;i<functions.size();i++){
		for(int j=0;j<functions.size();j++){
			if (j==i || is_deleted[j])
				continue;
			if (do_set_inclusion(functions[i]->variables(),functions[j]->variables(),less_than_comparator_variable)){
				functions[j]->product(*functions[i]);
				is_deleted[i]=true;
				break;
			}
		}
	}
	for(int j=0,i=0;j<functions.size();j++){
		if (!is_deleted[j]){
			new_functions.push_back(functions[j]);
			functions[j]->id()=i++;
		}
	}
	functions=new_functions;
	cerr<<"Num functions after processing = "<<functions.size()<<endl;
	cerr<<"mult-factor = "<<mult_factor<<endl;
	for (int i=0;i<functions.size();i++){
		functions[i]->normalize();
	}
	mult_factor=Double(1);
}

void GM::removeIrrelevantNetwork(vector<int>& evidence)
{

	vector<bool> processed(variables.size());
	set<int> unprocessed_nodes;
	//set<int> all_unprocessed_nodes;
	set<int> relevant_nodes;
	if(type==BAYES)
	{
		Graph graph;
		graph.makeDirectedGraph(this);

		for(int i=0;i<evidence.size();i++)
		{
			unprocessed_nodes.insert(evidence[i]);
			//all_unprocessed_nodes.insert(evidence[i]);
			relevant_nodes.insert(evidence[i]);
		}
		while(!unprocessed_nodes.empty())
		{
			int v=*(unprocessed_nodes.begin());

			unprocessed_nodes.erase(v);
			for(list<int>::const_iterator i=graph.getParentList()[v].begin();i!=graph.getParentList()[v].end();i++)
			{
				if(!processed[*i] ){
					relevant_nodes.insert(*i);
					unprocessed_nodes.insert(*i);
				}
			}
			processed[v]=true;
			//cout<<unprocessed_nodes.size()<<" "<<flush;

		}
		cerr<<"Number of Relevant nodes = "<<relevant_nodes.size()<<" out of "<<variables.size()<<endl;
	}
	else
	{
		for(int i=0;i<processed.size();i++)
		{
			processed[i]=true;
			relevant_nodes.insert(i);
		}
	}
	mult_factor=Double(1.0);
	vector<Function*> new_functions;
	for(int i=0;i<functions.size();i++)
	{
		bool deleted=false;
		for(int j=0;j<functions[i]->variables().size();j++)
		{
			if(!processed[functions[i]->variables()[j]->id()])
			{
				deleted=true;
				break;
			}
		}
		if(deleted)
		{
			delete(functions[i]);
			continue;
		}
		functions[i]->removeEvidence();
		if(functions[i]->variables().empty())
		{
			mult_factor*=functions[i]->table()[0];
			delete(functions[i]);
			continue;
		}
		new_functions.push_back(functions[i]);
	}
	functions=new_functions;
	vector<Variable*> new_variables;
	int count=0;
	for(int i=0;i<variables.size();i++)
	{
		if(processed[i] && variables[i]->value()==INVALID_VALUE)
		{
			new_variables.push_back(variables[i]);
			variables[i]->id()=count;
			count++;
		}
		else
		{
			variables[i]->id()=INVALID_VALUE;
		}
	}
	variables=new_variables;
        cout<<" Mult factor = "<<mult_factor<<endl;

}
