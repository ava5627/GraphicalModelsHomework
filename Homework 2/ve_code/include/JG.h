#ifndef JG_H_
#define JG_H_

#include "Globals.h"
#include "GM.h"

using namespace std;

struct JGNode;
struct JGEdge;

// The base class for any discrete join graph node
// The propagation algorithm uses the functions addFunction() 

struct JGNode {
protected:
	vector<Variable*> variables_;
	vector<JGEdge*> edges_;
	int id_;
	bool deleted_;
public:
	// Default constructor
	JGNode() :
		id_(INVALID_VALUE), deleted_(false) {
	}
	// Access to some internal variables
	bool& deleted() {
		return deleted_;
	}
	virtual ~JGNode() {
	}
	vector<Variable*>& variables() {
		return variables_;
	}
	vector<JGEdge*>& edges() {
		return edges_;
	}
	//JGNode* neighbor(int i);
	int& id() {
		return id_;
	}
	// Main functions which are to be overloaded by various architectures

	virtual void addFunction(Function& function) {
	}
	virtual void getCF(vector<Variable*>& cond_variables,
			Variable* marg_variable, CPT& cpt) {
	}
	virtual void updateVariables() {
	}
	virtual void getMarginal(vector<Variable*>& marg_variables,
			Function& function) {
	}
	virtual void initialize() {
	}
};

// Join Graph Node using the Shenoy Shafer architecture
struct JGNodeSS: public JGNode {
public:
	vector<Function*> original_functions;
	vector<Function*> functions;
	void compileAllFunctions(vector<Function*>& all_functions);
public:
	JGNodeSS() :
		JGNode() {
	}
	~JGNodeSS() {
		for (int i = 0; i < functions.size(); i++) {
			if (functions[i]) {

					delete (functions[i]);

			}
		}
	}
	void updateVariables();
	void getCF(vector<Variable*>& cond_variables, Variable* marg_variable,
			CPT& cf);
	void addFunction(Function& function);
	//void getMarginal(vector<Variable*>& marg_variables,vector<Double>& marg_table);
	void getMarginal(vector<Variable*>& marg_variables, Function& function);
	void initialize();
};


// Base class for Join Graph Edge
struct JGEdge {
protected:
	JGNode* node1_;
	JGNode* node2_;
	mutable vector<Variable*> variables_;
	Function* node1_to_node2_message_;
	Function* node2_to_node1_message_;
public:
	// Access to internal data structure
	JGEdge() :
		node1_to_node2_message_(new Function()), node2_to_node1_message_(
				new Function()) {
	}
	virtual ~JGEdge() {
		/*
		if (node1_to_node2_message_)
			delete (node1_to_node2_message_);
		if (node2_to_node1_message_)
			delete (node2_to_node1_message_);
			*/

	}
	virtual void initialize() {
	}
	JGNode* node1() {
		return node1_;
	}
	JGNode* node2() {
		return node2_;
	}
	vector<Variable*>& variables() const {
		return variables_;
	}
	Function& message1() {
		return *node1_to_node2_message_;
	}
	Function& message2() {
		return *node2_to_node1_message_;
	}

	// Functions

	void printMessages();
	virtual void sendMessage1to2() {
	}
	virtual void sendMessage2to1() {
	}
};

struct JGEdgeSS: public JGEdge {
protected:
	JGNodeSS* ss_node1_;
	JGNodeSS* ss_node2_;
public:
	JGEdgeSS() :
		JGEdge() {
	}
	~JGEdgeSS() {
		if (node1_to_node2_message_)
			delete (node1_to_node2_message_);
		if (node2_to_node1_message_)
			delete (node2_to_node1_message_);
	}
	JGEdgeSS(JGNodeSS* ss_node1__, JGNodeSS* ss_node2__);
	void initialize();
	void sendMessage1to2();
	void sendMessage2to1();
};

typedef enum {
	SS, LS, HSS, LSS, SSC, LSC, HSSC
} JG_TYPE;
struct JG {
private:
	int num_iterations_;
	int max_cluster_size;
	int i_bound_;
	//void reduce();
	static JGNode* addNode(JG_TYPE type = SS);
	JGEdge* addEdge(JGNode* s1, JGNode* s2, JG_TYPE type = SS);
	vector<JGNode*> marginal_nodes;
	GM* copy_of_gm;
public:
	vector<Function> marginals;
	vector<JGNode*> nodes;
	void printGraph(ostream& out);
	JG(GM& gm, int i_bound, int num_iterations, vector<int>& order,
			JG_TYPE type = SS);
	~JG() {
		// Get the edges
		set<JGEdge*> all_edges;
		for (auto & node : nodes) {
			for (int k = 0; k < node->edges().size(); k++) {
				all_edges.insert(node->edges()[k]);
			}
		}
		for(auto all_edge : all_edges){
			if (all_edge){
				delete all_edge;
			}
		}
		for (auto & node : nodes) {
			if (node) {
				delete node;
			}
		}

	}
	//void contractEdge(JGEdge* edge);
	//void putFunctions(GM& gm);
	//void minimize();
	int i_bound() {
		return i_bound_;
	}
	bool propagate();
	bool propagateDFS();
	bool propagate_Residual();
	void print(ostream& out);
	void clear();

	bool convergence_test();
	void updateMarginals(bool recompute=false);
};

#endif
