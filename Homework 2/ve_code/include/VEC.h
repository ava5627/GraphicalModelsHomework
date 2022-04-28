/*
 * VEC.h
 *
 *  Created on: Jun 29, 2010
 *      Author: Vibhav Gogate
 *      Email: vgogate@cs.washington.edu
 *      University of Washington, Seattle
 *      All rights reserved.
 */

#ifndef VEC_H_
#define VEC_H_

#include <Globals.h>
#include <Timer.h>
#include <GM.h>
#include <BE.h>
#include <JT.h>

struct VEC_NODE {
private:
	Variable* v;
	vector<bool> domain;
	int n;
public:
	VEC_NODE() :
			v(NULL), domain(vector<bool>()), n(-1) {
	}
	VEC_NODE(Variable* v_) :
			v(v_) {
		v->value() = INVALID_VALUE;
		domain = vector<bool>(v->domain_size());
		n = domain.size();
		for (int i = 0; i < domain.size(); i++) {
			domain[i] = true;
		}
	}
	void removeValue(int j) {
		assert((int) j < domain.size());
		if (domain[j]) {
			n--;
			domain[j] = false;
		}
	}
	void reset() {
		v->value() = INVALID_VALUE;
		domain = vector<bool>(v->domain_size());
		n = domain.size();
		for (int i = 0; i < domain.size(); i++) {
			domain[i] = true;
		}
	}
	int getN() {
		return n;
	}
	bool setNextValue() {
		int j = v->value();
		if (j == INVALID_VALUE)
			j = v->domain_size();
		while (1) {
			j--;
			//if ((int) j == domain.size())
			if (j < 0)
				return false;
			if (domain[j]) {
				v->value() = j;
				return true;
			}
		}
		return false;
	}
};
void print_assignment(vector<Variable*>& vars) {
	for (int i = 0; i < vars.size(); i++) {
		cout << vars[i]->id() << "=" << vars[i]->value() << ",";
	}
	cout << endl;
}


void VEC(GM& gm, vector<int>& order, vector<int>& sampling_order,
		clock_t& time_bound, ostream& out) {
	Timer timer;
	timer.start();
	vector<Variable*> cond_variables(sampling_order.size());
	for (int i = 0; i < sampling_order.size(); i++)
		cond_variables[i] = gm.variables[sampling_order[i]];
	Double1 log10_num_values = 0;
	for (int i = 0; i < cond_variables.size(); i++) {
		log10_num_values += log10((double) cond_variables[i]->domain_size());
	}
	//int num_values = Variable::getDomainSize(cond_variables);
	Double1 pe=0.0;

	//cerr << "log 10 num values = " << log10_num_values << endl;

	int n = cond_variables.size();
	int a[n];
	int f[n + 1];
	int o[n];
	int m[n];
	// Initialize
	for (int i = 0; i < n; i++) {
		cond_variables[i]->value() = 0;
		a[i] = 0;
		f[i] = i;
		o[i] = 1;
		m[i] = cond_variables[i]->domain_size();
	}
	f[n] = n;
	bool has_timed_out = false;
	Double1 num_explored = 0.0;
	while (1) {
		bool has_sol = true;
		if (timer.timeout(time_bound)) {
			has_timed_out = true;
			break;
		}
		num_explored += (1.0);


        BE be(gm.variables, gm.functions, order);
        //cout<<be.pe<<endl;
        pe += be.pe;

		int j = f[0];
		f[0] = 0;
		if (j == n)
			break;
		a[j] = a[j] + o[j];
		cond_variables[j]->value() = a[j];
		if (a[j] == 0 || a[j] == (m[j] - 1)) {
			o[j] = -o[j];
			f[j] = f[j + 1];
			f[j + 1] = j + 1;
		}
	}
	//cout << "Num-explored = " << num_explored << endl;
	//cerr << "Explored = " << num_explored << " out of "
	//		<< pow(10, log10_num_values) << endl;
	if (has_timed_out) {
		Double1 log10_factor = log10_num_values - log10(num_explored);
		out << log10(pe * gm.mult_factor) + log10_factor << endl;
		cerr << "Approximate log10(Z) = "<<log10(pe * gm.mult_factor) + log10_factor << endl;
	} else {
		out << log10(pe * gm.mult_factor) << endl;
		cerr << "Exact log10(Z) = "<<log10(pe * gm.mult_factor) << endl;
		//cerr << "Exact answer\n";
	}
}
void VEC_MAR(GM& gm, vector<int>& order, vector<int>& sampling_order,
		vector<set<int> >& clusters, clock_t& time_bound, ostream& out) {
    cout << "VEC MAR\n";
    Timer timer;
    timer.start();
    vector<Variable *> cond_variables(sampling_order.size());
    vector<bool> is_sampled(gm.variables.size());
    for (int i = 0; i < sampling_order.size(); i++) {
        cond_variables[i] = gm.variables[sampling_order[i]];
        is_sampled[cond_variables[i]->id()] = true;
    }
    Double1 log10_num_values = 0;
    for (int i = 0; i < cond_variables.size(); i++) {
        log10_num_values += log10((double) cond_variables[i]->domain_size());
    }
    vector<vector<Double1> > marginals(gm.variables.size());
    for (int i = 0; i < marginals.size(); i++) {
        marginals[i] = vector<Double1>(gm.variables[i]->domain_size(), 0);
    }
    Double1 pe = 0, mpe_value = 0;
    cerr << "log 10 num values = " << log10_num_values << endl;

    JT jt(gm.variables, gm.functions, clusters, order);
    int n = cond_variables.size();
    int a[n];
    int f[n + 1];
    int o[n];
    int m[n];
    // Initialize
    for (int i = 0; i < n; i++) {
        cond_variables[i]->value() = 0;
        a[i] = 0;
        f[i] = i;
        o[i] = 1;
        m[i] = cond_variables[i]->domain_size();
    }
    f[n] = n;
    bool has_timed_out = false;
    Double1 num_explored = 0.0;
    bool first = true;
    Double1 num_consistent = 0;
    while (1) {
        if (timer.timeout(time_bound)) {
            has_timed_out = true;
            break;
        }
        num_explored += (1.0);

        num_consistent += 1.0;
        BE be(gm.variables, gm.functions, order);
        //pe+=be.pe.toDouble1();
        pe += be.pe;
        if (be.pe > mpe_value) {
            mpe_value = be.pe;
        }

        if ((log10(mpe_value) - log10(be.pe)) < (Double1) 10) {
            jt.propagate();
            for (int ii = 0; ii < jt.marginals.size(); ii++) {
                if (!is_sampled[ii]) {
                    for (int jj = 0; jj < jt.marginals[ii].table().size();
                         jj++) {
                        marginals[ii][jj] += jt.marginals[ii].table()[jj]
                                             * be.pe;
                    }
                } else {
                    marginals[ii][gm.variables[ii]->value()] += be.pe;
                }
            }
        } else {
            for (int ii = 0; ii < jt.marginals.size(); ii++)
                if (is_sampled[ii])
                    marginals[ii][gm.variables[ii]->value()] += be.pe;
            cerr << "not propagating\n";
        }

        int j = f[0];
        f[0] = 0;
        if (j == n)
            break;
        a[j] = a[j] + o[j];
        cond_variables[j]->value() = a[j];
        if (a[j] == 0 || a[j] == (m[j] - 1)) {
            o[j] = -o[j];
            f[j] = f[j + 1];
            f[j + 1] = j + 1;
        }
    }
    cerr << "Explored = " << num_explored << " out of "
         << pow(10, log10_num_values) << endl;
    for (int i = 0; i < cond_variables.size(); i++)
        cond_variables[i]->value() = INVALID_VALUE;
    if (has_timed_out) {
        cout << "PE = " << log10(pe * gm.mult_factor) << endl;
        gm.printMarginalsUAI10(marginals, out);
    } else {
        cout << "PE = " << log10(pe * gm.mult_factor) << endl;
        gm.printMarginalsUAI10(marginals, out);
        cerr << "Exact answer\n";
    }
}
#endif /* VEC_H_ */
