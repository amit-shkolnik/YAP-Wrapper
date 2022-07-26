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


def update_label(graph, node_idx, parent_idx, new_label):
    graph.nodes[node_idx]["arc_label"]["new"] = new_label
    graph.nodes[node_idx]["parent"] = parent_idx
    graph.add_edge(parent_idx, node_idx, label=new_label)


def reorganize_conjunction(graph, conj_node_idx, conj_node, conj_parent_idx, conjuncts):
    graph.remove_edge(conj_node_idx, conjuncts[0])
    update_label(graph, conjuncts[0], conj_parent_idx, conj_node["arc_label"]["old"])
    for conjunct in conjuncts[1:]:
        graph.remove_edge(conj_node_idx, conjunct)
        # this should be conj also in SPMRL but there are occasional errors:
        update_label(graph, conjunct, conjuncts[0], "conj")
    graph.remove_edge(conj_parent_idx, conj_node_idx)
    update_label(graph, conj_node_idx, conjuncts[-1], "cc")
    return graph

def reverse_arc_direction(graph, node_idx, self_label, marker_label):
    parents = list(graph.predecessors(node_idx)) # idx of prepmod
    grandparent = list(graph.predecessors(parents[0]))[0] # idx of head
    graph.remove_edge(grandparent, parents[0])
    graph.remove_edge(parents[0], node_idx)
    update_label(graph=graph, node_idx=node_idx, parent_idx=grandparent, new_label=self_label)
    update_label(graph=graph, node_idx=parents[0], parent_idx=node_idx, new_label=marker_label)
    return graph


def convert_labels(graph, node_idx, node, parents):
    children = list(graph.successors(node_idx))
    entire_row_changes = CONVERSION_TABLE["labels_with_features_changed"]
    old_label = node['arc_label']['old']
    if node["pos"]["old"] == "CONJ": # the cc can by of any label.
        graph = reorganize_conjunction(graph, node_idx, node, parents[0], children)
    if node["arc_label"]["old"] == "pobj":
        graph = reverse_arc_direction(graph, node_idx, "obl", marker_label="case")
    elif node["arc_label"]["old"] == "ccomp":
        graph = reverse_arc_direction(graph, node_idx, "ccomp", marker_label="mark")
    elif node["arc_label"]["old"] == "advcl":
        graph = reverse_arc_direction(graph, node_idx, "advcl", marker_label="mark")
    elif node["arc_label"]["old"] == "relcomp":
        graph = reverse_arc_direction(graph, node_idx, "acl:relcl", marker_label="mark")
    if old_label == "neg":
        node["arc_label"]["new"] = "det" if graph.nodes[parents[0]]["pos"]["old"] == "NN" else "advmod"

    if old_label in LABELS.keys():
        update_label(graph, node_idx, parents[0], LABELS[old_label])

    elif old_label in entire_row_changes.keys():
        update_label(graph, node_idx, parents[0], entire_row_changes[old_label]["label"])
        node["pos"]["new"] = entire_row_changes[old_label]["pos"]
        node["features"]["new"] = compose_features(old_label, node, entire_row_changes)
    return graph


def convert_features(node):
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
            if node["pos"]["new"] == "PRON":
                if node["lemma"]["old"] == "עצמו":
                    if node["arc_label"]["old"] == "nn":
                        node["features"]["new"] += f"{node['features']['new']}|PronType=Emp|Reflex=Yes" if node["features"]["new"] else "PronType=Emp|Reflex=Yes"
                    else:
                        node["features"]["new"] += f"{node['features']['new']}|PronType=Prs|Reflex=Yes" if node["features"]["new"] else "PronType=Emp|Reflex=Prs"
                elif "PronType" not in node["features"]["new"]:
                    for prontype, lemmas in CONVERSION_TABLE["determiner_types"].items():
                        if node["lemma"]["old"] in lemmas:
                            node["features"]["new"] = f"{node['features']['new']}|PronType=Emp" if node["features"]["new"] else prontype


def specific_feats_conversions(node):
    if node["arc_label"]["old"] == "posspmod":
        node["features"]["new"] = 'Case=Gen'
    elif node['pos']['old'] == "PRP":
        if node['arc_label']['old'] == "subj":
            node["features"]["new"] += "|PronType=Prs"
        elif node['arc_label']['old'] == "det":
            node["features"]["new"] += "|PronType=Dem"
    return node


def convert_pos(graph, node_idx, node):
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
    if replacement_feats["method"] == "+feats+":
        features = f"{replacement_feats['addition'][0]}|{node['features']['old']}|{replacement_feats['addition'][1]}"
    elif replacement_feats["method"] == "feats+":
        if node['features']['old']:
            features = f"{node['features']['old']}|{replacement_feats['addition'][0]}"
        else:
            features = replacement_feats['addition'][0]
    elif replacement_feats["method"] == "+feats":
        features = f"{replacement_feats['addition'][0]}|{node['features']['old']}"
    else:
        features = replacement_feats["addition"][0]
    return features


def convert_graph(graph: nx.DiGraph) -> nx.DiGraph:
    for node_idx in graph.nodes:
        node = graph.nodes[node_idx]
        parents = list(graph.predecessors(node_idx))
        graph = convert_pos(graph, node_idx, node)
        convert_features(node)
        graph = convert_labels(graph, node_idx, node, parents)
        for att in ["word", "pos", "lemma", "arc_label"]:
            if not node[att]["new"]:
                node[att]["new"] = node[att]["old"]
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


def convert_dep_tree_to_ud(dep_tree: pd.DataFrame) -> pd.DataFrame:
    graph = convert_sentence_to_graph(dep_tree)
    nodelist = list(graph.nodes(data=True))
    df = pd.DataFrame(columns=["num", "word", "lemma", "pos", "features", "dependency_arc", "head"])
    for node in nodelist:
        df = df.append(
            {
                "num": str(node[0]+1),
                "word": node[1]["word"]["new"],
                "lemma": node[1]["lemma"]["new"],
                "pos": node[1]["pos"]["new"],
                "features":node[1]["features"]["new"],
                "dependency_arc": node[1]["arc_label"]["new"],
                "head": str(node[1]["parent"]+1)
        }, ignore_index=True)
    return df


if __name__ == '__main__':
    from yap_api import YapApi
    import matplotlib.pyplot as plt
    text = "האיש שהכרתי ברח"
    # text = "גנן גידל דגן בגן"
    ip = '127.0.0.1:8000'
    yap = YapApi()
    parsed_text = yap.run(text, ip)
    _dep_tree, _md_lattice, _ma_lattice, _segmented_text, _lemmas = yap.parse_sentence(text, ip)

    tree = convert_sentence_to_graph(_dep_tree)

    pos = nx.spring_layout(tree)

    nx.draw_networkx_nodes(tree, pos, cmap=plt.get_cmap('jet'), node_size=500, label="word")
    nx.draw_networkx_labels(tree, pos)
    nx.draw_networkx_edge_labels(tree, pos)
    nx.draw_networkx_edges(tree, pos, edgelist=tree.edges, arrows=True)
    plt.show()

