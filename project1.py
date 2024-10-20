'''
import sys
import networkx as nx
from collections import defaultdict
from math import log
from scipy.special import gammaln
import pandas as pd
import itertools
import random

# Function to precompute the unique values for each node/column
def get_node_value_ranges(df):
    value_ranges = {}
    for column in df.columns:
        unique_values = df[column].unique()
        value_ranges[column] = unique_values
    return value_ranges

# Function to precompute counts m_ijk for all nodes and parent configurations
def precompute_counts(dag, df, value_ranges):
    counts_dict = defaultdict(int)  # m_ijk counts for each node and parent configuration
    for node in dag.nodes:
        parents = list(dag.predecessors(node))
        parent_combinations = list(itertools.product(*[value_ranges[parent] for parent in parents]))
        
        for parent_config in parent_combinations:
            # Get the possible values for this node (child node)
            child_values = value_ranges[node]
            
            for child_value in child_values:
                counts_dict[(node, parent_config, child_value)] = compute_count_for_config(df, node, parent_config, child_value, parents)
    return counts_dict

# Function to compute the Bayesian score for any Bayesian network structure
def compute_bayesian_score(dag, counts_dict, alpha_ijk, value_ranges):
    total_score = 0.0
    # Log prior P(G) can be set as 0 or a fixed value depending on the assumption for the graph structure
    log_prior = 0  # Placeholder for log P(G), assuming uniform prior for simplicity
    total_score += log_prior

    # Iterate over all nodes in the graph
    for node in dag.nodes:
        # Get parents of the node
        parents = list(dag.predecessors(node))
        
        # If the node has no parents, there is only 1 parent configuration (independent node)
        if not parents:
            parent_combinations = [()]
        else:
            # Get all possible parent configurations (Cartesian product of parent values)
            parent_combinations = list(itertools.product(*[value_ranges[parent] for parent in parents]))
        
        # Iterate over all parent configurations
        for parent_config in parent_combinations:
            # Sum counts for all possible values of the node (child values)
            child_values = value_ranges[node]
            
            # First term: log(Gamma(alpha_ij0) / Gamma(alpha_ij0 + m_ij0))
            m_ij0 = sum(counts_dict[(node, parent_config, child_value)] for child_value in child_values)
            alpha_ij0 = sum(alpha_ijk[node][parent_config][child_value] for child_value in child_values)

            # Ensure these terms are handled correctly for unconnected nodes
            if alpha_ij0 > 0 or m_ij0 > 0:  # Avoid adding zeros unnecessarily
                total_score += gammaln(alpha_ij0) - gammaln(alpha_ij0 + m_ij0)
            
            # Second summation: log(Gamma(alpha_ijk + m_ijk) / Gamma(alpha_ijk))
            for child_value in child_values:
                m_ijk = counts_dict[(node, parent_config, child_value)]
                alpha_ijk_value = alpha_ijk[node][parent_config][child_value]

                total_score += gammaln(alpha_ijk_value + m_ijk) - gammaln(alpha_ijk_value)

    return total_score

# Function to compute counts for a specific node, parent configuration, and value of child node
def compute_count_for_config(df, node, parent_config, child_value, parents):
    count = 0
    # Iterate through the dataframe and count how many rows match the given configuration
    for idx, row in df.iterrows():
        match = True
        # Check if the parent values match the parent_config
        for p_idx, parent in enumerate(parents):
            if row[parent] != parent_config[p_idx]:
                match = False
                break
        # Check if the node value matches the child_value
        if match and row[node] == child_value:
            count += 1
    return count

# Function to update only the affected parts of the Bayesian score when the DAG changes
def update_score_on_change(dag, counts_dict, alpha_ijk, affected_node, value_ranges, df):
    parents = list(dag.predecessors(affected_node))
    parent_combinations = list(itertools.product(*[value_ranges[parent] for parent in parents]))
    score_update = 0.0

    # Recompute counts for the affected node
    for parent_config in parent_combinations:
        for child_value in value_ranges[affected_node]:
            counts_dict[(affected_node, parent_config, child_value)] = compute_count_for_config(df, affected_node, parent_config, child_value, parents)

    # Recompute the score contribution for the affected node
    for parent_config in parent_combinations:
        for child_value in value_ranges[affected_node]:
            m_ijk = counts_dict[(affected_node, parent_config, child_value)]
            alpha_ijk_value = alpha_ijk[affected_node][parent_config][child_value]
            score_update += gammaln(alpha_ijk_value + m_ijk) - gammaln(alpha_ijk_value)

        total_parent_count = sum(counts_dict[(affected_node, parent_config, child_value)] for child_value in value_ranges[affected_node])
        total_alpha_count = sum(alpha_ijk[affected_node][parent_config][child_value] for child_value in value_ranges[affected_node])
        score_update += gammaln(total_alpha_count) - gammaln(total_alpha_count + total_parent_count)

    return score_update

# Function to write the graph to a file
def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    """
    Compute the optimal Bayesian network using the K2 algorithm with multiple trials of different node orders
    and a higher limit on the number of parents per node.
    
    :param infile: Input CSV file containing the dataset.
    :param outfile: Output file to write the resulting Bayesian network.
    :param max_parents: The maximum number of parents allowed per node.
    :param num_trials: The number of trials to run with different node orders.
    """
    max_parents=3
    num_trials=50
    # Step 1: Load data
    df = pd.read_csv(infile)
    
    # Step 2: Precompute node value ranges (unique values for each node)
    value_ranges = get_node_value_ranges(df)

    # Track the best network structure and score found so far
    best_dag = None
    best_score = float('-inf')

    # Step 3: Run multiple trials with different node orders
    node_orders = []
    
    # Use the default column order for the first trial
    node_orders.append(df.columns.tolist())

    # Use the reversed column order for the second trial
    node_orders.append(list(reversed(df.columns.tolist())))

    # Generate random orders for the remaining trials
    for _ in range(num_trials - 2):  # Minus 2 since we already have two trials
        random_order = df.columns.tolist()
        random.shuffle(random_order)  # Shuffle the order randomly
        node_orders.append(random_order)
    
    # Step 4: Run K2 for each node order
    for trial, node_order in enumerate(node_orders):
        print(f"\nRunning trial {trial + 1}/{len(node_orders)} with node order: {node_order}")
        
        # Step 5: Set up an empty DAG (directed acyclic graph) with no edges
        dag = nx.DiGraph()
        dag.add_nodes_from(node_order)
        
        # Step 6: Initialize parameters for K2 (e.g., alpha_ijk as unit Dirichlet prior with alpha_ijk = 1)
        alpha_ijk = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 1)))
        
        # Step 7: Precompute initial counts
        counts_dict = precompute_counts(dag, df, value_ranges)

        # Step 8: Initialize the Bayesian score for the initial structure (no edges)
        current_score = compute_bayesian_score(dag, counts_dict, alpha_ijk, value_ranges)
        
        # Step 9: K2 Algorithm - Greedily add parents to maximize the score
        for i, node in enumerate(node_order):
            current_parents = []
            best_local_score = current_score
            
            # Consider nodes earlier in the order as potential parents
            for j in range(i):
                candidate_parent = node_order[j]
                
                if len(current_parents) < max_parents:
                    # Temporarily add the candidate parent to the graph
                    dag.add_edge(candidate_parent, node)
                    
                    # Recompute the counts and the score update after adding this edge
                    score_update = update_score_on_change(dag, counts_dict, alpha_ijk, node, value_ranges, df)
                    
                    # If the new score is better, keep the edge, otherwise remove it
                    if score_update > best_local_score:
                        best_local_score = score_update
                        current_parents.append(candidate_parent)
                    else:
                        dag.remove_edge(candidate_parent, node)
            
            # Update the current score after processing this node
            current_score = best_local_score

        # Step 10: Check if the current trial resulted in a better score
        if current_score > best_score:
            best_score = current_score
            best_dag = dag.copy()  # Save the best DAG found so far

    # Step 11: Create idx2names mapping (map node names to themselves)
    idx2names = {name: name for name in df.columns.tolist()}
    
    # Step 12: Write the best DAG to the output file
    write_gph(best_dag, idx2names, outfile)
    
    # Step 13: Print the optimal parent-child relationships
    print("\nOptimal Parent-Child Relationships (Best Trial):")
    for node in df.columns.tolist():
        parents = list(best_dag.predecessors(node))
        if parents:
            print(f"{node} has parents: {', '.join(parents)}")
        else:
            print(f"{node} has no parents")
    
    # Step 14: Print the final Bayesian network score
    print(f"\nFinal Bayesian Network Score (Best Trial): {best_score:.4f}")



def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()

'''
import sys
import networkx as nx
import pandas as pd
import numpy as np

def read_data(infile):
    """Reads the input CSV file and returns the data as a pandas DataFrame."""
    return pd.read_csv(infile)

def bayesian_score(dag, data, node, parents):
    """Computes the Bayesian score of a node with given parents."""
    if not parents:
        # If no parents, compute the marginal probability of the node
        counts = data[node].value_counts()
        total_count = len(data)
        score = 0
        for count in counts:
            score += count * np.log(count / total_count)
        return score
    else:
        # Compute the conditional probability of node given its parents
        parent_data = data[parents]
        score = 0
        # For each unique combination of parent values
        for _, parent_combination in parent_data.drop_duplicates().iterrows():
            mask = (parent_data == parent_combination).all(axis=1)
            sub_data = data[mask]
            total_count = len(sub_data)
            counts = sub_data[node].value_counts()
            for count in counts:
                score += count * np.log(count / total_count)
        return score

def k2_search(data, var_order, max_parents):
    """Performs the K2 algorithm for Bayesian network structure learning."""
    num_vars = len(var_order)
    dag = nx.DiGraph()
    dag.add_nodes_from(var_order)

    for i in range(1, num_vars):
        current_node = var_order[i]
        parent_set = []
        best_score = bayesian_score(dag, data, current_node, parent_set)
        while len(parent_set) < max_parents:
            candidates = [var_order[j] for j in range(i) if var_order[j] not in parent_set]
            if not candidates:
                break

            best_candidate = None
            best_candidate_score = float('-inf')
            for candidate in candidates:
                new_parents = parent_set + [candidate]
                score_with_candidate = bayesian_score(dag, data, current_node, new_parents)
                if score_with_candidate > best_candidate_score:
                    best_candidate_score = score_with_candidate
                    best_candidate = candidate

            if best_candidate_score > best_score:
                parent_set.append(best_candidate)
                dag.add_edge(best_candidate, current_node)
                best_score = best_candidate_score
            else:
                break
    return dag

def write_gph(dag, filename):
    """Writes the resulting DAG to the output file."""
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(edge[0], edge[1]))

def compute(infile, outfile):
    """Fits a Bayesian network to the data using the K2 algorithm."""
    data = read_data(infile)
    var_order = data.columns.tolist()  # Variable ordering
    max_parents = 2  # Set a limit on the number of parents per node

    dag = k2_search(data, var_order, max_parents)
    
    write_gph(dag, outfile)

    print("Learned Bayesian Network Structure:")
    for edge in dag.edges():
        print(f"{edge[0]} -> {edge[1]}")

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)

if __name__ == '__main__':
    main()
