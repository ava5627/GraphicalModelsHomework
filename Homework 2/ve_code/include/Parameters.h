/*
 * Parameters.h
 *
 *  Created on: Jun 28, 2010
 *      Author: Vibhav Gogate
 *      Email: vgogate@cs.washington.edu
 *      University of Washington, Seattle
 *      All rights reserved.
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

struct UAI2010Parameters {
	// The graphical model
	GM& gm;
	// Time information
	clock_t& time_bound;
	//clock_t& memory_bound;

	// Task type
	string& task;

	// The best treewidth order
	vector<int> bt_order;
	// The best w-cutset order
	vector<int> bw_order;
	// The sampling order
	vector<int> s_order;
	vector<set<int> > bt_clusters;
	vector<set<int> > bw_clusters;
	int num_iterations;
	// i-bound, rb-bound and max-restarts
	int i_bound;
	double rb_bound;
	int max_restarts;
	int max_domain_size;
	void adjust_i_bound() {
		i_bound = (int) floor(log(2) * (double) i_bound / log(max_domain_size));
	}
	UAI2010Parameters(GM& gm_, clock_t& curr_time_, double& total_memory, string& task_) :
			gm(gm_), time_bound(curr_time_), task(task_), max_restarts(1), num_iterations(
					5), i_bound(1), rb_bound(1), max_domain_size(0) {
		double estimate;
		max_domain_size = 0;
		int eff_var_size = 0;
		for (auto & variable : gm.variables) {
			if (max_domain_size < (int) variable->domain_size())
				max_domain_size = variable->domain_size();
			if (variable->value() == INVALID_VALUE) {
				eff_var_size++;
			}
		}
		clock_t ord_time_bound = 120;
		if ((int) time_bound < (int) 50) {
			if (gm.mode == DET) {
				rb_bound = log((double) (1 << 22));
			} else
				rb_bound = log((double) (1 << 24)+5000);
			i_bound = 12;
			adjust_i_bound();
			ord_time_bound = 2;
		} else {
			num_iterations = 45;
			i_bound = 17;
			adjust_i_bound();
			rb_bound = log((double) (1 << 30));
		}
		cout << "i-bound = " << i_bound << endl;
		cout << "rb-bound = " << rb_bound << endl;
		int smallest_max_cluster_size = gm.variables.size();
		vector<int> curr_order, curr_s_order;
		vector < set<int> > curr_clusters;
		int curr_max_cluster_size;
		double curr_cutset_size;
		double smallest_cutset_size = DBL_MAX;
		Timer timer;
		timer.start();
		for (int i = 0; i < 500000; i++) {
			curr_order.clear();
			curr_clusters.clear();
			if (timer.timeout(ord_time_bound)) {
				cout<<" number of orderings considered ="<< i<<endl;
				break;
			}
			gm.getMinFillOrdering(curr_order, curr_clusters,
					estimate, curr_max_cluster_size);
			//cout<<"m = "<<curr_max_cluster_size<<endl;
			if (curr_max_cluster_size < smallest_max_cluster_size) {
				cerr<<"Cluster size changed from "<<smallest_max_cluster_size<<" to "<<curr_max_cluster_size<<endl;
				smallest_max_cluster_size = curr_max_cluster_size;
				bt_clusters = curr_clusters;
				bt_order = curr_order;
			}
			for (int j = 0; j < 1; j++) {
				gm.rearrangeOrdering_randomized(curr_order, curr_clusters,
						curr_s_order, rb_bound);
				//gm.rearrangeOrdering(curr_order, curr_clusters, curr_s_order, rb_bound);

				curr_cutset_size = 0;
				for (int k : curr_s_order) {
					curr_cutset_size += log(
							gm.variables[k]->domain_size());
				}
				//cout<<"\t "<<curr_cutset_size<<endl;
				if (curr_cutset_size < smallest_cutset_size) {
					//cout << curr_cutset_size << " " << smallest_cutset_size
						//	<<" "<<curr_s_order.size()<< " " << i << " " << j << endl;
					cout<<"Changed from "<<smallest_cutset_size<<" to "<<curr_cutset_size<<endl;
					smallest_cutset_size = curr_cutset_size;
					bw_order = curr_order;
					bw_clusters = curr_clusters;
					s_order = curr_s_order;
				}
				if (timer.timeout(ord_time_bound)) {
					cout<<" number of orderings considered ="<< i<<endl;
					break;
				}
			}
		}
		cerr << "Treewidth = " << smallest_max_cluster_size << endl;
		cerr << "# sampled = " << s_order.size() << endl;
	}
	bool exact_inf_test() {
		double how_many = 0.0;
		double limit=1.2;
		if ((int) time_bound < 50) {
			limit = 1;
		}
		/*
		if (gm.mode==DET){
			limit=10;
		}
		*/
		//cerr << "Limit for exact inf = " << limit << endl;
		for (int var : s_order) {
				how_many += log10(gm.variables[var]->domain_size());
			if (how_many > limit) {
				return false;
			}
		}
		return true;
	}
};

#endif /* PARAMETERS_H_ */
