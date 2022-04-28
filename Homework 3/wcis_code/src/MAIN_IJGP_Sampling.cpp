//
// Created by Vibhav Gogate on 3/21/22.
//

#include <iostream>
#include "GM.h"
#include "RB_IJGP_Sampler.h"


using namespace std;

int help()
{
    cerr << "Usage: wcis [options] <uai-filename> <order-filename> <w-cutset-vars-filename> <time-in-seconds> <is-quality>\n";
    cerr << "-------------------------------------------------------------------------------------\n";
    cerr << "\t The five arguments are required\n";
    cerr << "\t The program will output an estimate of the partition function\n";
    cerr << "-------------------------------------------------------------------------------------\n";
    //cerr << " \t The following four options help you control the quality of the proposal distribution (i and n),\n";
    //cerr << " \t and the size of the w-cutset (w)\n";
    //cerr << " \t -i [int] : The i-bound of IJGP, i >=0 \n";
    //cerr << " \t -w [int] : The w-cutset bound, w >=0 \n";
    //cerr << " \t -n [int] : Number of iterations for IJGP, n>=0\n";
    exit(1);
}
string uaifilename;
string orderfilename;
string cutsetfilename;
int i_bound=1,w=0,n_iter=10;
clock_t total_time = 100;
int internal_option=0;
void readParameters(int argc, char* argv[])
{
    if (argc < 6) {
        help();
        exit(0);
    }
    for(int i=1;i<argc-6;i++){
        string mystring=argv[i];
        if (mystring == "-i"){
            i_bound=atoi(argv[++i]);
        }
        else if (mystring == "-w"){
            w=atoi(argv[++i]);
        }
        else if (mystring == "-n"){
            n_iter=atoi(argv[++i]);
        }
        else if (mystring == "-d"){
            internal_option=atoi(argv[++i]);
        }
        else{
            cerr<<"ERROR ------ Wrong options\n";
            help();
            exit(-1);
        }
    }
    uaifilename = argv[argc - 5];
    orderfilename = argv[argc - 4];
    cutsetfilename=argv[argc-3];
    total_time=atoi(argv[argc-2]);
    i_bound=atoi(argv[argc-1]);
}
void read_cutset_vars(vector<int>& s_order){
    s_order=vector<int>();
    // Read the cutset vars
    ifstream in2(cutsetfilename.c_str());
    if (in2.good()) {
        int num_cutset_vars;
        in2>>num_cutset_vars;
        s_order=vector<int>(num_cutset_vars);
        for (int j = 0; j < num_cutset_vars; j++) {
            if(in2.good()) in2 >> s_order[j];
            else{
                cerr<<"Error while reading cutset file\n";
                exit(-1);
            }
        }
    } else{
        cerr<<"Cutset file not found\n";
        exit(1);
    }
    in2.close();
}

void write_cutset_vars(vector<int>& s_order){
    ofstream  out(cutsetfilename);
    out<<s_order.size();
    for (int j = 0; j < s_order.size(); j++) {
        out<<" "<<s_order[j];
    }
    out<<endl;
    out.close();
}


void read_order(vector<int>& order, int num_variables)
{
    // Read the order
    order=vector<int> (num_variables);
    ifstream in(orderfilename.c_str());
    if (in.good()) {
        for (int j = 0; j < num_variables; j++) {
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
}
void write_order(vector<int>& order){
    ofstream  out(orderfilename);
    for (int j = 0; j < order.size(); j++) {
        out<<" "<<order[j];
    }
    out<<endl;
    out.close();
}

int main(int argc, char* argv[])
{

    readParameters(argc, argv);
    GM gm;
    gm.readUAI08(uaifilename.c_str());
    cout<<"Using:\n";
    cout<<"\t\t UAI file = "<<uaifilename<<endl;
    cout<<"\t\t Order file = "<<orderfilename<<endl;
    cout<<"\t\t wcutset file = "<<cutsetfilename<<endl;
    cout<<"\t\t Time Bound = "<<total_time<<endl;
    cout<<"\t\t Proposal Quality = "<<i_bound<<endl;
    vector<int> s_order,order;
    if (internal_option!=0){

        double estimate;

        // The best treewidth order
        vector<int> bt_order;
        // The best w-cutset order
        vector<int> bw_order;
        // The sampling order
        vector<int> mys_order;
        vector<set<int> > bt_clusters;
        vector<set<int> > bw_clusters;
        gm.getMinFillOrdering(bt_order, bt_clusters, estimate);
        bw_order = bt_order;
        bw_clusters = bt_clusters;
        gm.getWCutset(bw_order, bw_clusters, mys_order, w);
        order=bt_order;
        s_order=mys_order;
        write_cutset_vars(s_order);
        write_order(order);
    }
    else {
        read_cutset_vars(s_order);
        read_order(order, gm.variables.size());
        if (w > 0) {
            if (!gm.verifyWCutset(order, s_order, w)) {
                cerr << "W-cutset is incorrect; Exiting\n";
                exit(-1);
            }
        }
    }


    Timer timer;
    timer.start();
    JG jg(gm,i_bound,n_iter,order,LSS);
    jg.propagate();
    total_time -= timer.elapsed_seconds();
    RB_IJGP_Sampler sampler;
    sampler.computePE(gm, jg, 0, order, s_order, total_time);
    return 0;
}

