import pandas as pd
import networkx as nx
import json
from collections import defaultdict
import logging

logger = logging.getLogger()
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)


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
    parents = list(graph.predecessors(node_idx))  # idx of prepmod
    try:
        grandparent = list(graph.predecessors(parents[0]))[0]  # idx of head
    except:
        sentence = " ".join([graph.nodes[n]["form"]["old"] for n in graph.nodes])
        node_word = graph.nodes[node_idx]["form"]["old"]
        parent_label = graph.nodes[parents[0]]["arc_label"]["old"]

        logger.warning(f"The parse for sentence {sentence} may be wrong. "
                       f"The parent of {node_word} should be '{marker_label}' but it's '{parent_label}'. "
                       f"Aborting edges swap.")
        return graph
    graph.remove_edge(grandparent, parents[0])
    graph.remove_edge(parents[0], node_idx)
    update_label(graph=graph, node_idx=node_idx, parent_idx=grandparent, new_label=self_label)
    update_label(graph=graph, node_idx=parents[0], parent_idx=node_idx, new_label=marker_label)
    return graph


def convert_labels(graph, node_idx, node, parents):
    children = list(graph.successors(node_idx))
    entire_row_changes = CONVERSION_TABLE["labels_with_features_changed"]
    old_label = node['arc_label']['old']
    if not node["arc_label"]["new"]:
        if old_label in LABELS.keys():
            update_label(graph, node_idx, parents[0], LABELS[old_label])

        elif old_label in entire_row_changes.keys():
            update_label(graph, node_idx, parents[0], entire_row_changes[old_label]["label"])
            node["pos"]["new"] = entire_row_changes[old_label]["pos"]
            node["features"]["new"] = compose_features(old_label, node, entire_row_changes)
    if node["pos"]["old"] == "CONJ":  # the cc can by of any label.
        graph = reorganize_conjunction(graph, node_idx, node, parents[0], children)
    if node["arc_label"]["new"] == "obl":
        graph = reverse_arc_direction(graph, node_idx, "obl", marker_label="case")
    elif node["arc_label"]["new"] == "ccomp":
        graph = reverse_arc_direction(graph, node_idx, "ccomp", marker_label="mark")
    elif node["arc_label"]["new"] == "advcl":
        graph = reverse_arc_direction(graph, node_idx, "advcl", marker_label="mark")
    elif node["arc_label"]["new"] == "acl:relcl":
        graph = reverse_arc_direction(graph, node_idx, "acl:relcl", marker_label="mark")
    elif node["arc_label"]["new"] == "obj":
        graph = reverse_arc_direction(graph, node_idx, "obj", marker_label="case:acc")
    if old_label == "neg":
        node["arc_label"]["new"] = "det" if graph.nodes[parents[0]]["pos"]["old"] == "NN" else "advmod"

    return graph


def base_features_conversion(old_features):
    feats = old_features.split("|")
    if all(f in feats for f in ["gen=F", "gen=M"]):
        # otherwise both Gender=Fem and Gender=Masc will be added
        feats.append("gen=F|gen=M")
        feats.remove("gen=F")
        feats.remove("gen=M")
    current_feats = [f for f in feats if not f.startswith("suf_")]
    suffix_feats = [f.replace("suf_", "") for f in feats if f.startswith("suf_")]
    old_features = "|".join(current_feats)
    for feature in current_feats:
        if feature:
            old_features = old_features.replace(feature, FEATURES[feature])
    return old_features, suffix_feats


def convert_features(graph, node_idx, node):
    old_features = node["features"]["old"]
    converted_features, suffix_feats = base_features_conversion(old_features)
    if not node["features"]["new"]:
        if suffix_feats:
            if node["pos"]["old"] == 'NN':
                node["lemma"]["new"] = '_'
                node["pos"]["new"] = 'NOUN'
                node["features"]["new"] = f"Definite=Def|{converted_features}"
            else:
                node["features"]["new"] = converted_features
        else:
            node["features"]["new"] = converted_features
            node = specific_feats_conversions(node)
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
    return graph

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
            graph.nodes[node_idx - 1]["features"]["new"] = 'Case=Acc' # probably redundant
        elif graph.nodes[node_idx-1]['lemma']['old'] == 'של':
            graph.nodes[node_idx - 1]["features"]["new"] = 'Case=Gen'
        elif graph.nodes[node_idx-1]['pos']['old'] == 'IN':
            graph.nodes[node_idx - 1]["features"]["new"] = '_'
        node['form']['new'] = "_" + node['lemma']['old']
        node['pos']['new'] = 'PRON'
    else:
        old_pos = node['pos']['old']
        if old_pos in POS.keys():
            node['pos']['new'] = POS[old_pos]
        elif old_pos in ENTIRE_LINE.keys():
            node['pos']['new'] = ENTIRE_LINE[old_pos]["pos"]
            node['features']['new'] = compose_features(old_pos, node, ENTIRE_LINE)
            node['arc_label']['new'] = ENTIRE_LINE[old_pos].get("arc_label") or ""
            concat = ENTIRE_LINE[old_pos].get("concat")
            if concat == "before":
                node['form']['new'] = f"_{node['form']['old']}"
            elif concat == "after":
                node['form']['new'] = f"{node['form']['old']}_"
    return graph


def compose_features(key, node, map):
    replacement_feats = map[key]["feats"]
    new_base_features, suffix_feats = base_features_conversion(node['features']['old'])
    if replacement_feats["method"] == "+feats+":
        features = f"{replacement_feats['addition'][0]}|{new_base_features}|{replacement_feats['addition'][1]}"
    elif replacement_feats["method"] == "feats+":
        if node['features']['old']:
            features = f"{new_base_features}|{replacement_feats['addition']}"
        else:
            features = replacement_feats['addition']
    elif replacement_feats["method"] == "+feats":
        features = f"{replacement_feats['addition']}|{new_base_features}"
    else:
        features = replacement_feats["addition"]
    return features


def convert_graph(graph: nx.DiGraph) -> nx.DiGraph:
    for node_idx in graph.nodes:
        node = graph.nodes[node_idx]
        parents = list(graph.predecessors(node_idx))
        graph = convert_pos(graph, node_idx, node)
        graph = convert_features(graph, node_idx, node)
        graph = convert_labels(graph, node_idx, node, parents)
        for att in ["form", "pos", "lemma", "arc_label"]:
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
        clean_suffix_feats, _ = base_features_conversion(suffix_feats.replace("suf_", ""))
        if pos in ["NN", "NN_S_PP", "S_PP"]:
            graph.add_node(
                node_idx+1,
                form={"new": "_של_", "old": ""},
                pos={"new": "ADP", "old": ""},
                lemma={"new": "_של_", "old": ""},
                features={"new": "_", "old": features},
                arc_label={"new": "case:gen", "old": "case:gen"},
                parent=node_idx+2,
            )
            graph.add_node(
                node_idx+2,
                form={"new": PRONOUNS[suffix_feats], "old": ""},
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
                form={"new": PRONOUNS[suffix_feats], "old": ""},
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
                       form={"new": "", "old": row["word"]},
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
    df = pd.DataFrame(columns=["num", "form", "lemma", "pos", "features", "dependency_arc", "head"])
    for node in nodelist:
        df = df.append(
            {
                "num": str(node[0]+1),
                "form": node[1]["form"]["new"],
                "lemma": node[1]["lemma"]["new"],
                "pos": node[1]["pos"]["new"],
                "features":node[1]["features"]["new"],
                "dependency_part": node[1]["arc_label"]["new"],
                "head": str(node[1]["parent"]+1)
        }, ignore_index=True)
    return df


if __name__ == '__main__':
    from yap_api import YapApi
    import matplotlib.pyplot as plt
    text = "הוא הוריד את שכרם למינימום"
    # text = "עכשיו אני מרגיש כאילו לא יודע כלום עכשיו אני מחיש את צעדיי היא מסתכלת בחלון רואה אותי עובר בחוץ היא לא יודעת מה עובר עליי."
    ip = '127.0.0.1:8000'
    yap = YapApi()
    parsed_text = yap.run(text, ip)
    _dep_tree, _md_lattice, _ma_lattice, _segmented_text, _lemmas = yap.parse_sentence(text, ip)
    print(_dep_tree[["word", "dependency_part", "empty", "dependency_arc"]])
    tree = convert_sentence_to_graph(_dep_tree)
    df = convert_dep_tree_to_ud(_dep_tree)
    print(df[["form", "dependency_part", "features", "head"]])
    # pos = nx.spring_layout(tree)
    #
    # nx.draw_networkx_nodes(tree, pos, cmap=plt.get_cmap('jet'), node_size=500, label="form")
    # nx.draw_networkx_labels(tree, pos)
    # nx.draw_networkx_edge_labels(tree, pos)
    # nx.draw_networkx_edges(tree, pos, edgelist=tree.edges, arrows=True)
    # plt.show()

