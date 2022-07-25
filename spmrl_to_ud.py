import pandas as pd
import networkx as nx
import json
from collections import defaultdict

with open('conversion_table.json', 'r') as f:
    CONVERSION_TABLE = json.load(f)

PRONOUNS = CONVERSION_TABLE["pronouns"]
POS = CONVERSION_TABLE["basic_pos"]
LABELS = CONVERSION_TABLE["basic_labels"]
FEATURES = CONVERSION_TABLE["basic_features"]
ENTIRE_LINE = CONVERSION_TABLE["entire_line_pos_conversion"]


def reorganize_conjunction(graph, node_idx):
    node = graph.nodes[node_idx]
    top_label = node["pos"]["old"]
    children = [n for n in graph.nodes if graph.nodes[n]["parent"] == node_idx]
    min_node = graph.nodes[children[0]]
    min_node["label"]["new"] = top_label


def reverse_arc_direction(graph, node_idx, new_head_label):
    node = graph.nodes[node_idx]
    parent = list(graph.predecessors(node_idx))
    head_children = [n for n in graph.successors(node_idx) if graph.nodes[n]["arc_label"]["old"] == new_head_label]
    if head_children:
        graph.remove_edge(parent[0], node_idx)
        graph.remove_edge(node_idx, head_children[0])
        graph.add_edge(parent[0], head_children[0], label="obl")
        graph.add_edge(head_children[0], node_idx, label="case")


def convert_labels(graph, node_idx):
    node = graph.nodes[node_idx]
    parent_idx = list(graph.predecessors(node_idx))
    entire_row_changes = CONVERSION_TABLE["labels_with_features_changed"]
    if node["arc_label"]["old"] == "prepmod":
        reverse_arc_direction(graph, node_idx, "pobj")
    old_label = node['arc_label']['old']
    if parent_idx:
        parent = graph.nodes[parent_idx[0]]
        if old_label == "neg":
            node["arc_label"]["new"] = "det" if parent["pos"]["old"] == "NN" else "advmod"

    if old_label in LABELS.keys():
        node['arc_label']['new'] = LABELS[old_label]
    elif old_label in entire_row_changes.keys():
        node["arc_label"]["new"] = entire_row_changes[old_label]["label"]
        node["pos"]["new"] = entire_row_changes[old_label]["pos"]
        node["features"]["new"] = compose_features(old_label, node, entire_row_changes)


def convert_features(graph, node_idx):
    node = graph.nodes[node_idx]
    old_features = node["features"]["old"]

    noun_feats = "|".join([x for x in old_features.split("|") if 'suf' not in x])
    if not node["features"]["new"]:
        if 'suf_' in old_features:
            if node["pos"]["old"] == 'NN':
                node["lemma"]["new"] = '_'
                node["pos"]["new"] = 'NOUN'
                node["features"]["new"] = f"Definite=Def|{noun_feats}|xx_UD=Seg"
            else:
                feats = old_features.replace("suf_", "").split("|")
                for feature in feats:
                    if feature:
                        old_features = old_features.replace(feature, FEATURES[feature])
                node["features"]["new"] = old_features
        else:
            feats = old_features .split("|")
            for feature in feats:
                if feature:
                    old_features = old_features.replace(feature, FEATURES[feature])
            node["features"]["new"] = old_features
            specific_feats_conversions(node)


def specific_feats_conversions(node):
    if node["arc_label"]["old"] == "posspmod":
        node["features"]["new"] = 'Case=Gen'
    elif node['pos']['old'] == "PRP":
        if node['arc_label']['old'] == "subj":
            node["features"]["new"] += "|PronType=Prs"
        elif node['arc_label']['old'] == "det":
            node["features"]["new"] += "|PronType=Dem"
    return node


def convert_pos(graph, node_idx):
    node = graph.nodes[node_idx]
    if node['pos']['old'] == 'S_PRN':
        if graph.nodes[node_idx-1]['pos']['old'] == 'AT':
            graph.nodes[node_idx - 1]["features"]["new"] = 'Case=Acc'
        elif graph.nodes[node_idx-1]['lemma']['old'] == 'של':
            graph.nodes[node_idx - 1]["features"]["new"] = 'Case=Gen'
        elif graph.nodes[node_idx-1]['pos']['old'] == 'IN':
            graph.nodes[node_idx - 1]["features"]["new"] = ''
        node['word']['new'] = "_" + node['lemma']['old']

    else:
        old_pos = node['pos']['old']
        if old_pos in POS.keys():
            node['pos']['new'] = POS[old_pos]
        else:
            if old_pos:
                node['pos']['new'] = ENTIRE_LINE[old_pos]["pos"]
                node['features']['new'] = compose_features(old_pos, node, ENTIRE_LINE)
                node['arc_label']['new'] = ENTIRE_LINE[old_pos].get("arc_label") or ""
    return graph


def compose_features(key, node, map):
    replacement_feats = map[key]["feats"]
    if replacement_feats["old"] == "+feats+":
        features = replacement_feats["new"][0] + node['features']['old'] + replacement_feats["new"][1]
    elif replacement_feats["old"] == "feats+":
        features = node['features']['old'] + replacement_feats["new"][0]
    elif replacement_feats["old"] == "+feats":
        features = replacement_feats["new"][0] + node['features']['old']
    else:
        features = replacement_feats["new"][0]
    return features


def convert_graph(graph: nx.DiGraph) -> nx.DiGraph:
    for node_idx in graph.nodes:
        convert_pos(graph, node_idx)
        convert_features(graph, node_idx)
        convert_labels(graph, node_idx)
    return graph


def add_functional_nodes_if_necessary(graph, node_idx):
    node = graph.nodes[node_idx]
    pos = node["pos"]["old"]
    features = node["features"]["old"]
    offset = 0
    if "suf_" in features:
        suffix_feats = "|".join([x for x in features.split("|") if 'suf' in x])
        clean_suffix_feats = "|".join([x.replace("suf_", "") for x in features.split("|") if 'suf' in x])
        if pos in ["NN", "NN_S_PP", "S_PP"]:
            graph.add_node(
                node_idx+1,
                word={"new": "_של_", "old": ""},
                pos={"new": "ADP", "old": ""},
                lemma={"new": "_של_", "old": ""},
                features={"new": "_", "old": features},
                arc_label={"new": "case:gen", "old": "case:gen"},
                parent=node_idx+2,
            )
            graph.add_node(
                node_idx+2,
                word={"new": PRONOUNS[suffix_feats], "old": ""},
                pos={"new": "PRON", "old": ""},
                lemma={"new": "הוא", "old": ""},
                features={"new": f"Case=Gen|{clean_suffix_feats}|PronType=Prs", "old": ""},
                arc_label={"new": "nmod:poss", "old": "nmod:poss"},
                parent=node_idx,
            )
            offset += 2
        elif pos in ["DTT", "DT"]: # כולנו
            # add another row of הוא
            graph.add_node(
                node_idx+1,
                word={"new": PRONOUNS[suffix_feats], "old": ""},
                pos={"new": "PRON", "old": ""},
                lemma={"new": "הוא", "old": ""},
                features={"new": f"Case=Gen|{clean_suffix_feats}|PronType=Prs", "old": features},
                arc_label={"new": "nmod:poss", "old": "nmod:poss"},
                parent=node_idx,
            )
            offset += 1
    return offset


def add_edges(graph, parents_traceback):
    for x in range(len(graph.nodes)):
        node = graph.nodes[x]
        if node["parent"] == -1:
            continue
        elif node["parent"] in parents_traceback.keys():
            parent = parents_traceback[node["parent"]]
            graph.add_edge(u_of_edge=parent, v_of_edge=x, label=node["arc_label"]["old"])
        else:
            graph.add_edge(u_of_edge=node["parent"], v_of_edge=x, label=node["arc_label"]["old"])


def add_nodes(graph, dep_tree):
    offset = 0
    parents_traceback = defaultdict(int)
    for i, row in dep_tree.iterrows():
        parent = int(row["dependency_arc"])-1
        if parent >= 0 and offset > 0:  # do not add offset if parent is 0 (root)
            parents_traceback[parent] = parent + offset
        graph.add_node(i+offset,
                       word={"new": "", "old": row["word"]},
                       pos={"new": "", "old": row['pos']},
                       lemma={"new": "", "old": row['lemma']},
                       features={"new": "", "old": row["empty"]},
                       arc_label={"new": "", "old": row["dependency_part"]},
                       parent=parent,
                       )
        offset += add_functional_nodes_if_necessary(graph, i+offset)
    return parents_traceback


def convert_sentence_to_graph(dep_tree) -> nx.DiGraph:
    graph = nx.DiGraph()
    parents_traceback = add_nodes(graph, dep_tree)
    add_edges(graph, parents_traceback)
    convert_graph(graph)
    return graph


if __name__ == '__main__':
    from yap_api import YapApi
    import matplotlib.pyplot as plt
    # text = "הם מצאו דרך לעשות קופה"
    text = "גנן גידל דגן בגנו"
    ip = '127.0.0.1:8000'
    yap = YapApi()
    parsed_text = yap.run(text, ip)
    _dep_tree, _md_lattice, _ma_lattice, _segmented_text, _lemmas = yap.parse_sentence(text, ip)

    tree = convert_sentence_to_graph(_dep_tree)
    for n in tree.nodes:
        print(tree.nodes[n])

    pos = nx.spring_layout(tree)
    nx.draw_networkx_nodes(tree, pos, cmap=plt.get_cmap('jet'), node_size=500, label="word")
    nx.draw_networkx_labels(tree, pos)
    nx.draw_networkx_edge_labels(tree, pos)
    nx.draw_networkx_edges(tree, pos, edgelist=tree.edges, arrows=True)
    plt.show()

