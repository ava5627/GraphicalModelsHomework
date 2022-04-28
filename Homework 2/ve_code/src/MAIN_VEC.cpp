/*
 * MAIN_IJGP.cpp
 *
 *  Created on: Jun 23, 2010
 *      Author: Vibhav Gogate
 *      Email: vgogate@cs.washington.edu
 *      University of Washington, Seattle
 *      All rights reserved.
 */

#include <iostream>
#include <fstream>
#include "BE.h"
#include "GM.h"
#include "Timer.h"
#include "Parameters.h"
#include "JT.h"
#include "VEC.h"
//#include <Map.h>
using namespace std;

string getOutFilename(string& str) {
	int d = -1;
	for (int i = str.size() - 1; i > -1; i--) {
		if (str.data()[i] == '/') {
			d = i;
			break;
		}
	}
	return str.substr(d + 1, str.size());
}

string outfilename = "out";
string uaifilename;
string evidfilename;
vector<vector<pair<int, int> > > evidences;
int seed;
string task;
int help() {
	cerr
			<< "Usage: vec-2014 [options] <uaifilename> <evidfilename> <queryfilename> <task>\n";
	cerr << "\t The four arguments are required\n";
	cerr << "\t <task> can be PR or MAR\n";
	cerr << "\t query file name will be ignored\n";
	cerr
			<< "\t The output will be stored in <uaifilename>.task in the working directory\n";
	exit(1);
}
void readParameters(int argc, char* argv[]) {
	if (argc < 4) {
		help();
	}
	uaifilename = argv[argc - 4];
	evidfilename = argv[argc - 3];
	seed = 1000022838L;
	task = argv[argc - 1];
	outfilename = getOutFilename(uaifilename) + "." + task;
	cout << outfilename << endl;
	ofstream out(outfilename.c_str());
	out << task << endl;
	out.close();
}
void readEvidence() {

	ifstream in(evidfilename.c_str());
	int num_evidence = 1;
	if (in.good()) {
		//in >> num_evidence;
		evidences = vector<vector<pair<int, int> > >(num_evidence);
		int i = 0;
		int curr_num_evidence;
		in >> curr_num_evidence;
		if (curr_num_evidence == 0) {
			num_evidence = 0;
			evidences.clear();
		}
		for (int j = 0; j < curr_num_evidence; j++) {
			int var, val;
			in >> var >> val;
			//gm.variables[var]->value() = val;
			evidences[i].push_back(pair<int, int>(var, val));
		}
	}
	cout << "Evidence read\n";
	in.close();
}

int old_main(int argc, char* argv[]) {

	Timer timer;
	timer.start();
	char* uai_time = getenv("INF_TIME");
	clock_t total_time;
	if (uai_time) {
		cout << "Read total-time = " << uai_time << endl;
		total_time = atoi(uai_time);
	} else {
		total_time = 20;
	}
	//total_time = (int) (((double) total_time - 2) * (double) 0.95);
	cerr << "Time-bound =" << total_time << endl;
	char* uai_memory = getenv("INF_MEMORY");
	double total_memory = 2;
	if (uai_memory) {
		total_memory = atof(uai_memory);
	}
	cerr << "Memory bound =" << total_memory << " GB"<<endl;
	total_memory *= 10E7;
	total_time--;
	readParameters(argc, argv);
	GM gm;
	gm.readUAI08(uaifilename.c_str());
	readEvidence();

	int num_evidences = 1;
	if (!evidences.empty()) {
		num_evidences = evidences.size();
	}
	total_time -= timer.elapsed_seconds();

	cout << "Total time = " << total_time << endl;

	ofstream out(outfilename.c_str(), ios::app);
	//out << num_evidences << endl;
	vector<int> evidence;
	if (!evidences.empty()) {
		int num = 0;
		evidence = vector<int>(evidences[num].size());
		for (int i = 0; i < evidences[num].size(); i++) {
			evidence[i] = evidences[num][i].first;
			gm.variables[evidences[num][i].first]->value() =
					evidences[num][i].second;
		}
	}

	if (task == "PR") {
		gm.removeIrrelevantNetwork(evidence);
	} else if (task == "MAR") {
		gm.setEvidenceBeliefsUAI08(evidence);
	}
	if (gm.mode == DET) {
        cerr<<"Cannot handle zero probabilities\n";
        exit(-1);
	}
	// SampleSearch is very expensive. Therefore if the SAT instance is empty, set gm mode to non-deterministic
	UAI2010Parameters params(gm, total_time, total_memory, task);
	cerr << "# sampled = " << params.s_order.size() << endl;

	if (task == "PR") {
		total_time -= timer.elapsed_seconds();
        VEC(gm, params.bw_order, params.s_order, total_time, out);
		out.close();
		return 0;
	} else {
		total_time -= timer.elapsed_seconds();
        VEC_MAR(gm, params.bw_order, params.s_order, params.bw_clusters,
					total_time, out);
		out.close();
		return 0;
	}

}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr<<"Usage: vec <uaifile> <orderfile>\n";
        exit(-1);
    }
    uaifilename = argv[argc-2];
    string orderfilename=argv[argc-1];
    clock_t timelimit=1E08;
    GM gm;
    gm.readUAI08(uaifilename.c_str());
    vector<int> order(gm.variables.size());

    // Read the order
    ifstream in(orderfilename.c_str());
    if (in.good()) {
        for (int j = 0; j < gm.variables.size(); j++) {
            if(in.good()) in >> order[j];
            else{
                cerr<<"Error while reading order file\n";
                exit(-1);
            }
        }
    } else{
        cerr<<"Ordering file not found\n";
        exit(1);
    }
    in.close();

    /*
    double estimate;
    int max_cluster_size;
    vector < set<int> > clusters;
    //gm.getMinFillOrdering(order,clusters,estimate,max_cluster_size);
    gm.g
    ofstream order_out(orderfilename.c_str());
    for(int i=0;i<order.size();i++)
        order_out<<order[i]<<" ";
    order_out<<endl;
    order_out.close();
    */

    vector<int> s_order;
    string outfilename = "out";
    ofstream out(outfilename.c_str());

    VEC(gm, order,s_order, timelimit, out);
    out.close();

    return 1;

}