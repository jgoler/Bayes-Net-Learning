import sys
import networkx as nx
import pandas as pd
from math import log
from scipy.special import gammaln  # To compute the logarithm of the gamma function


def bayesian_score(node, parents, data, alpha_value):
    """
    Calculate the Bayesian score for a given node and its parent set.
    :param node: The node for which the score is computed
    :param parents: The parents of the node
    :param data: The dataset (in pandas DataFrame format)
    :param alpha_value: Uniform prior value for Dirichlet distribution
    :return: Bayesian score for the node
    """
    ri = len(data[node].unique())  # Number of unique values the node can take
    qi = 1
    for parent in parents:
        qi *= len(data[parent].unique())  # Product of unique values in the parent nodes

    score = 0

    # Group data by parent configurations
    if parents:
        parent_data = data.groupby(parents)
    else:
        parent_data = [([], data)]  # No parents case

    # Loop through each parent configuration
    for _, group in parent_data:
        mij = 0
        for k in range(ri):
            # Count for node in state k given parent configuration j
            mijk = group[group[node] == k].shape[0]  # Counts rows where node == k
            mij += mijk

            # Use uniform Dirichlet prior alpha_value
            alpha_ijk_current = alpha_value

            # Add to score using the formula from the image
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

    # Loop through all nodes in the graph
    for node in dag.nodes():
        # Get the parents of the current node
        parents = list(dag.predecessors(node))

        # Calculate the Bayesian score for the node given its parents
        node_score = bayesian_score(node, parents, data, alpha_value)

        # Add the node's score to the total score
        total_score += node_score

    return total_score



def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    # Load the dataset from infile (assuming CSV format)
    data = pd.read_csv(infile)

    # Create the DAG structure using NetworkX
    dag = nx.DiGraph()

    # Define the edges where the first item in each pair is the parent of the second item
    edges = [
        ('age', 'numparentschildren'),
        ('age', 'passengerclass'),
        ('age', 'numsiblings'),
        ('portembarked', 'fare'),
        ('portembarked', 'passengerclass'),
        ('fare', 'numparentschildren'),
        ('numparentschildren', 'sex'),
        ('numparentschildren', 'numsiblings'),
        ('passengerclass', 'sex'),
        ('passengerclass', 'survived'),
        ('sex', 'survived')
    ]

    # Add these edges to the graph
    dag.add_edges_from(edges)

    # Set up uniform Dirichlet prior (alpha value)
    uniform_prior_value = 1  # This is the value for the uniform prior

    # Calculate total Bayesian score for the entire graph
    total_score = calculate_total_bayesian_score(dag, data, uniform_prior_value)
    print(f"Total Bayesian score for the graph: {total_score}")

    # Write the DAG to the output file
    # Here, the idx2names mapping is straightforward, as we're using node names directly
    idx2names = {node: node for node in dag.nodes()}  # Maps each node to itself

    write_gph(dag, idx2names, outfile)



def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()

