import sys
import pandas as pd
from math import log
from scipy.special import gammaln 
import networkx as nx
import pprint


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

# initialize the counts for the entire graph
def initialize_counts(df, parents_dict):
    counts_dict = {}

    for node in df.columns:
        parents = parents_dict.get(node, [])
        if len(parents) == 0:
            # node has no parents --> just count the occurrences of node values
            counts_dict[node] = df[node].value_counts().to_dict()
        else:
            # Count occurrences of node values given parent instantiations
            parent_values = df[parents].drop_duplicates().values
            print(f"parent_values: {parent_values}")
            for p_vals in parent_values:
                subset = df[(df[parents] == p_vals).all(axis=1)]
                counts_dict[(node, tuple(p_vals))] = subset[node].value_counts().to_dict()
    return counts_dict

# compute the Bayesian Score for a given node
def compute_node_score(df, node, parents, counts_dict, alpha=1.0):
    score = 0
    unique_values = df[node].unique()

    if len(parents) == 0:
        # No parents, use the marginal likelihood
        for x_val, m_i in counts_dict.get(node, {}).items():
            score += gammaln(alpha + m_i) - gammaln(alpha)
        score += gammaln(alpha * len(unique_values)) - gammaln(alpha * len(unique_values) + len(df))
    
    else:
        # Node has parents --> use conditional likelihood instead
        parent_values = df[parents].drop_duplicates().values

        for p_vals in parent_values:
            # Convert parent values to Python integers to avoid type mismatches
            p_vals_tuple = tuple(int(p) for p in p_vals)

            # Ensure the key exists in counts_dict, dynamically compute counts if missing
            if (node, p_vals_tuple) not in counts_dict:
                # Recompute counts dynamically if they are missing
                subset = df[(df[parents] == p_vals).all(axis=1)]
                counts_dict[(node, p_vals_tuple)] = subset[node].value_counts().to_dict()
            
            subset_count = sum(counts_dict[(node, p_vals_tuple)].values())
            score += gammaln(alpha * len(unique_values)) - gammaln(alpha * len(unique_values) + subset_count)

            for x_val, m_ijk in counts_dict[(node, p_vals_tuple)].items():
                score += gammaln(alpha + m_ijk) - gammaln(alpha)
    
    return score


'''
# dynamic programming structure to update only relevant nodes
def update_score_on_graph_change(df, node, parents, counts_dict, score_dict, alpha=1.0):
    # recompute the score for the affected node and update the score dictionary
    new_score = compute_node_score(df, node, parents, counts_dict, alpha)
    score_dict[node] = new_score
'''
def update_score_on_graph_change(df, node, parents, counts_dict, score_dict, alpha=1.0):
    # Step 1: Recompute counts for the affected note based on its new parent configuration
    if len(parents) == 0:
        counts_dict[node] = df[node].value_counts().to_dict()
    else:
        # If there are parents, compute the conditional counts based on the new parent values
        parent_values = df[parents].drop_duplicates().values

        # initialize the new count structure for the node

        counts_dict[node] = {}

        # Iterate over each unique combination of parent values
        for p_vals in parent_values:
            # Convert parent values to a tuple (for consistent dictionary keying)
            p_vals_tuple = tuple(p_vals)

            # Extract the subset of data that matches the parent values
            subset = df[(df[parents] == p_vals).all(axis=1)]

            # Count the occurrences of the node values within this subset
            counts_dict[node][p_vals_tuple] = subset[node].value_counts().to_dict()
    
    # Step 2: Recompute the score for the affected node using the updated counts
    new_score = compute_node_score(df, node, parents, counts_dict, alpha)

    # Step 3: Update the score dictionary with the new score
    score_dict[node] = new_score

# compute the bayesian score of a given graph
def compute_bayesian_score(score_dict):
    total_score = sum(score_dict.values())
    return total_score

'''
def calculate_bayesian_score(df, parents_dict, alpha=1.0):
    score = 0
    for i, node in enumerate(df.columns):
        # Get parent nodes of the current node
        parents = parents_dict.get(node, [])
        if len(parents) == 0:
            # node has no parents --> compute the marginal
            unique_values = df[node].unique()
            for x_val in unique_values:
                m_i = len(df[df[node] == x_val]) # count occurences of value
                score += gammaln(alpha + m_i) - gammaln(alpha)
            score += gammaln(alpha * len(unique_values)) - gammaln(alpha * len(unique_values) + len(df))
        else:
            # node does have parents --> compute conditional probability
            parent_values = df[parents].drop_duplicates().values
            for p_vals in parent_values:
                subset = df[(df[parents] == p_vals).all(axis=1)]
                N_ij = len(subset)
                unique_node_values = df[node].unique()
                # add current node to score
                score += gammaln(alpha * len(unique_node_values)) - gammaln(alpha * len(unique_node_values) + N_ij)

                for x_val in unique_node_values:
                    m_ijk = len(subset[subset[node] == x_val])
                    score += gammaln(alpha + m_ijk) - gammaln(alpha)
    return score
'''


def k2(df, node_order, max_parents=3):
    G = nx.DiGraph() # we want it to be a directed graph (bayes nets are DAG)
    G.add_nodes_from(df.columns)

    for i, node in enumerate(node_order):
        parents = []
        

def compute(infile, outfile):
    # Read the input CSV file using pandas
    df = pd.read_csv(infile)
    parents_dict = {}
    
    # Initialize counts for all nodes
    counts_dict = initialize_counts(df, parents_dict)

    pprint.pprint(counts_dict)

    # Initialize score dictionary
    score_dict = {}
    for node in df.columns:
        parents = parents_dict.get(node, [])
        score_dict[node] = compute_node_score(df, node, parents, counts_dict, 1.0)


    total_score = compute_bayesian_score(score_dict)
    # Output the intial total score and return data structures for further updates
    print(f"Initial Bayesian Score: {total_score}")

    parents_dict = {
        'survived': ['age', 'fare', 'sex'],  # Survival might depend on age, fare, and sex
        'fare': ['passengerclass', 'age'],   # Fare might depend on passenger class and age
        'age': [],                           # Age has no parents (it’s an independent variable)
        'passengerclass': [],                # Passenger class is independent (no parents)
        'sex': [],                           # Sex is independent (no parents)
        'portembarked': ['passengerclass'],  # Port embarked might depend on passenger class
        'numsiblings': ['passengerclass'],   # Number of siblings might depend on passenger class
        'numparentschildren': ['passengerclass', 'age']  # Number of parents/children might depend on class and age
    }


    for node in df.columns:
        parents = parents_dict.get(node, [])
        score_dict[node] = update_score_on_graph_change(df, node, parents, counts_dict, score_dict, 1.0)

    total_score = compute_bayesian_score(score_dict)

    print(f"Second Bayesian Score: {total_score}")


    
    

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
