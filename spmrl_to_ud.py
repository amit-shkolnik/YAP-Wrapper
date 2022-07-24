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


def convert_labels(graph, node_idx):
    node = graph.nodes[node_idx]
    old_label = node['arc_label']['old']
    if old_label in LABELS.keys():
        node['pos']['new'] = LABELS[old_label]


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
                replacement_feats = ENTIRE_LINE[old_pos]["feats"]
                if replacement_feats["old"] == "+feats+":
                    features = replacement_feats["new"][0] + node['features']['old'] + replacement_feats["new"][1]
                elif replacement_feats["old"] == "feats+":
                    features = node['features']['old'] + replacement_feats["new"][0]
                elif replacement_feats["old"] == "+feats":
                    features = replacement_feats["new"][0] + node['features']['old']
                else:
                    features = replacement_feats["new"][0]
                node['features']['new'] = features
                node['arc_label']['new'] = ENTIRE_LINE[old_pos].get("arc_label") or ""
    return graph


def convert_graph(graph: nx.DiGraph) -> nx.DiGraph:
    for node_idx in graph.nodes:
        node = graph.nodes[node_idx]
        if node:
            convert_pos(graph, node_idx)
            convert_features(graph, node_idx)

    return graph


def add_empty_nodes_if_necessary(graph, i, word, pos, lemma, features, arc_label, parent):
    offset = 0
    if "suf_" in features:
        suffix_feats = "|".join([x for x in features.split("|") if 'suf' in x])
        clean_suffix_feats = "|".join([x.replace("suf_", "") for x in features.split("|") if 'suf' in x])
        if pos in ["NN", "NN_S_PP", "S_PP"]:
            graph.add_node(
                i+1,
                word={"new": "_של_", "old": ""},
                pos={"new": "ADP", "old": ""},
                lemma={"new": "_של_", "old": ""},
                features={"new": "_", "old": features},
                arc_label={"new": "case:gen", "old": ""},
                parent=i+2,
            )
            graph.add_node(
                i+2,
                word={"new": PRONOUNS[suffix_feats], "old": ""},
                pos={"new": "PRON", "old": ""},
                lemma={"new": "הוא", "old": ""},
                features={"new": f"Case=Gen|{clean_suffix_feats}|PronType=Prs", "old": ""},
                arc_label={"new": "nmod:poss", "old": ""},
                parent=i,
            )
            offset += 2
        elif pos in ["DTT", "DT"]: # כולנו
            # add another row of הוא
            graph.add_node(
                i+1,
                word={"new": PRONOUNS[suffix_feats], "old": ""},
                pos={"new": "PRON", "old": ""},
                lemma={"new": "הוא", "old": ""},
                features={"new": f"Case=Gen|{clean_suffix_feats}|PronType=Prs", "old": features},
                arc_label={"new": "nmod:poss", "old": ""},
                parent=i,
            )
            offset += 1
    return offset


def convert_sentence_to_graph(dep_tree) -> nx.DiGraph:
    graph = nx.DiGraph()
    offset = 0
    parents_traceback = defaultdict(int)
    for i, row in dep_tree.iterrows():
        word = row["word"]
        pos = row['pos']
        lemma = row['lemma']
        features = row["empty"]
        arc_label = row["dependency_part"]
        parent = int(row["dependency_arc"])
        if parent != 0 and offset > 0:  # do not add offset if parent is 0 (root)
            parents_traceback[parent] = parent + offset
        graph.add_node(i+offset,
                       word={"new": "", "old": word},
                       pos={"new": "", "old": pos},
                       lemma={"new": "", "old": lemma},
                       features={"new": "", "old": features},
                       arc_label={"new": "", "old": arc_label},
                       parent=parent,
                       )
        offset += add_empty_nodes_if_necessary(graph, i, word, pos, lemma, features, arc_label, parent)

    for x in range(len(graph.nodes)):
        node = graph.nodes[x]
        if node["parent"] == 0:
            # print(f"parent {parent}, child {i} word {word} label {arc_label}")
            graph.add_edge(u_of_edge=0, v_of_edge=x + 1, label='root')
        elif node["parent"] in parents_traceback.keys():
            # print(f"parent {parent}, child {i} word {word} label {arc_label}")
            graph.add_edge(u_of_edge=parents_traceback[node["parent"]], v_of_edge=x + 1, label=node["arc_label"]["old"])
        elif node["arc_label"]["old"] in ["prepmod", "pobj"]:
            if node["arc_label"]["old"] == "prepmod":
                pobj_node_idx, new_pobj_parent = reverse_arc_direction(graph, x, new_head_label="pobj")
                graph.add_edge(u_of_edge=new_pobj_parent, v_of_edge=pobj_node_idx, label="obl")
                graph.add_edge(u_of_edge=pobj_node_idx, v_of_edge=x+1, label="case")
        else:
            graph.add_edge(u_of_edge=node["parent"], v_of_edge=x + 1, label=node["arc_label"]["old"])
    convert_graph(graph)
    return graph


def reverse_arc_direction(graph, current_head_idx, new_head_label):
    current_head_node = graph.nodes[current_head_idx]
    for idx in range(current_head_idx, len(graph.nodes)):
        node = graph.nodes[idx]
        if node["arc_label"]["old"] == new_head_label:
            if node["parent"] == current_head_idx+1:
                required_node_idx = idx+1
                new_required_parent = current_head_node["parent"]
                return required_node_idx, new_required_parent


def specific_feats_conversions(node):
    if node["arc_label"]["old"] == "posspmod":
        node["features"]["new"] = 'Case=Gen'
    elif node['pos']['old'] == "PRP":
        if node['arc_label']['old'] == "subj":
            node["features"]["new"] += "|PronType=Prs"
        elif node['arc_label']['old'] == "det":
            node["features"]["new"] += "|PronType=Dem"
    return node


if __name__ == '__main__':
    from yap_api import YapApi
    import matplotlib.pyplot as plt

    text = "גנן גידל דגן בגן"
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

