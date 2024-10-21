import sys
import networkx as nx
import pandas as pd
from math import log
from scipy.special import gammaln  
import itertools
import random

class BayesianNetworkScorer:
    def __init__(self, dag, data, alpha_value):
        self.dag = dag
        self.data = data
        self.alpha_value = alpha_value
        self.node_scores = {}  
        
        self.precompute_node_scores()

    def precompute_node_scores(self):
        """
        Precompute the Bayesian scores for all nodes in the graph.
        This initializes the node_scores dictionary with the score for each node.
        """
        for node in self.dag.nodes():
            parents = list(self.dag.predecessors(node))
            self.node_scores[node] = bayesian_score(node, parents, self.data, self.alpha_value)

    def update_node_score(self, node):
        """
        Update the Bayesian score for a given node when its parents change.
        """
        parents = list(self.dag.predecessors(node))
        self.node_scores[node] = bayesian_score(node, parents, self.data, self.alpha_value)

    def calculate_total_bayesian_score(self):
        """
        Calculate the total Bayesian score by summing up the precomputed scores.
        """
        return sum(self.node_scores.values())

    def add_edge(self, parent, child):
        """
        Add an edge to the DAG and update the scores of the affected nodes.
        """
        self.dag.add_edge(parent, child)
        self.update_node_score(child)  

    def remove_edge(self, parent, child):
        """
        Remove an edge from the DAG and update the scores of the affected nodes.
        """
        self.dag.remove_edge(parent, child)
        self.update_node_score(child)  


def bayesian_score(node, parents, data, alpha_value):
    """
    Calculate the Bayesian score for a given node and its parent set.
    :param node: The node for which the score is computed
    :param parents: The parents of the node
    :param data: The dataset (in pandas DataFrame format)
    :param alpha_value: Uniform prior value for Dirichlet distribution
    :return: Bayesian score for the node
    """
    ri = len(data[node].unique()) 
    qi = 1
    for parent in parents:
        qi *= len(data[parent].unique())  

    score = 0

    if parents:
        parent_data = data.groupby(parents)
    else:
        parent_data = [([], data)]  

    for _, group in parent_data:
        mij = 0
        for k in range(ri):
            mijk = group[group[node] == k].shape[0] 
            mij += mijk

            alpha_ijk_current = alpha_value

            score += gammaln(alpha_ijk_current + mijk) - gammaln(alpha_ijk_current)

        alpha_ij0 = alpha_value * ri
        score += gammaln(alpha_ij0) - gammaln(alpha_ij0 + mij)

    return score


def calculate_total_bayesian_score(dag, data, alpha_value):
    """
    Calculate the total Bayesian score for the entire graph.
    The total score is the sum of the Bayesian scores of each node in the DAG.
    :param dag: The DAG representing the Bayesian network.
    :param data: The dataset (in pandas DataFrame format).
    :param alpha_value: Uniform prior value for Dirichlet distribution.
    :return: Total Bayesian score for the entire graph.
    """
    total_score = 0

    for node in dag.nodes():
        parents = list(dag.predecessors(node))
        node_score = bayesian_score(node, parents, data, alpha_value)
        total_score += node_score

    return total_score



def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def k2_algorithm(dag, data, alpha_value, node_order, max_parents):
    """
    K2 algorithm to find an optimal Bayesian network structure.
    :param dag: Initial empty DAG (should be a DiGraph from NetworkX)
    :param data: The dataset (in pandas DataFrame format).
    :param alpha_value: Uniform prior value for Dirichlet distribution.
    :param node_order: A list representing the order of the nodes.
    :param max_parents: Maximum number of parents each node can have.
    :return: The optimal DAG found by the K2 algorithm.
    """
    scorer = BayesianNetworkScorer(dag, data, alpha_value)

    for i, node in enumerate(node_order):
        current_parents = []
        best_score = scorer.node_scores[node]
        for potential_parent in node_order[:i]:
            if len(current_parents) < max_parents:
                dag.add_edge(potential_parent, node)
                scorer.update_node_score(node)  
                new_score = scorer.node_scores[node]
                if new_score > best_score:
                    best_score = new_score
                    current_parents.append(potential_parent)
                else:
                    dag.remove_edge(potential_parent, node)
                    scorer.update_node_score(node) 
    return dag


def compute(infile, outfile):
    """
    Run the K2 algorithm with multiple node orders to find the best DAG.
    :param infile: Input CSV file with data.
    :param outfile: Output file for writing the DAG.
    :param num_orders: Number of different node orders to try.
    """

    num_orders = 2000
    data = pd.read_csv(infile)

    if data.columns.duplicated().any():
        raise ValueError("Dataset contains duplicate column names. Please ensure all columns are uniquely named.")
    
    nodes = list(data.columns)

    uniform_prior_value = 1  # Can change if uniform prior isn't 1 if we had more certainty
    max_parents = 8  # play around with this

    best_score = float('-inf')
    best_dag = None
    best_node_order = None

    for _ in range(num_orders):
        node_order = random.sample(nodes, len(nodes))
        dag = nx.DiGraph()
        dag.add_nodes_from(node_order)

        current_dag = k2_algorithm(dag, data, uniform_prior_value, node_order, max_parents)

        scorer = BayesianNetworkScorer(current_dag, data, uniform_prior_value)
        current_score = scorer.calculate_total_bayesian_score()

        if current_score > best_score:
            best_score = current_score
            best_dag = current_dag
            best_node_order = node_order

    print(f"Best node order: {best_node_order}")
    print(f"Best Bayesian score: {best_score}")

    print("\nParent-child relationships in the best DAG:")
    for parent, child in best_dag.edges():
        print(f"{parent} -> {child}")

    idx2names = {node: node for node in best_dag.nodes()}  
    write_gph(best_dag, idx2names, outfile)

   





def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()

