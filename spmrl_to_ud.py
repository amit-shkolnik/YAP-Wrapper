import pandas as pd
from tqdm import tqdm
import networkx as nx
import json

with open('conversion_table.json', 'r') as f:
    CONVERSION_TABLE = json.load(f)

PRONOUNS = CONVERSION_TABLE["pronouns"]
POS = CONVERSION_TABLE["basic_pos"]
FEATURES = CONVERSION_TABLE["basic_features"]
entire_line_pos_conversion = CONVERSION_TABLE["entire_line_pos_conversion"]


def add_empty_nodes_if_necessary(nodes, i, word, pos, lemma, features, arc_label, parent):
    shift_by = 0
    if "suf_" in features and pos == "NN":
        suffix_feats = "|".join([x for x in features.split("|") if 'suf' in x])
        clean_suffix_feats = "|".join([x.replace("suf_", "") for x in features.split("|") if 'suf' in x])
        nodes.add_node(
            i+1,
            word={"new": "_של_", "old": "_"},
            pos={"new": "ADP", "old": "_"},
            lemma={"new": "_של_", "old": "_"},
            features={"new": "_", "old": features},
            arc_label={"new": "case:gen", "old": "_"},
            parent={"new": i+2, "old": parent},
        )
        nodes.add_node(
            i+2,
            word={"new": PRONOUNS[suffix_feats], "old": "_"},
            pos={"new": "PRON", "old": "_"},
            lemma={"new": "הוא", "old": "_"},
            features={"new": f"Case=Gen|{clean_suffix_feats}|PronType=Prs", "old": "_"},
            arc_label={"new": "nmod:poss", "old": "_"},
            parent={"new": i+1, "old": parent},
        )
        shift_by += 2

    return shift_by


def convert_sentence_to_graph(dep_tree) -> nx.DiGraph:
    nodes = nx.DiGraph()
    shift_by = 0
    for i, row in dep_tree.iterrows():
        word = row["word"]
        pos = row['pos']
        lemma = row['lemma']
        features = row["empty"]
        arc_label = row["dependency_part"]
        parent = int(row["dependency_arc"]) + shift_by if not 0 else int(row["dependency_arc"])
        print(word, pos, lemma, features, arc_label, parent)
        nodes.add_node(i+shift_by,
                       word={"new": "", "old": word},
                       pos={"new": "", "old": pos},
                       lemma={"new": "", "old": lemma},
                       features={"new": "", "old": features},
                       arc_label={"new": "", "old": arc_label},
                       parent={"new": "", "old": parent},
                       )
        shift_by += add_empty_nodes_if_necessary(nodes, i, word, pos, lemma, features, arc_label, parent)

        if parent == 0:
            # print(f"parent {parent}, child {i} word {word} label {arc_label}")
            nodes.add_edge(u_of_edge=0, v_of_edge=i + 1, label='root')
        else:
            # print(f"parent {parent}, child {i} word {word} label {arc_label}")

            nodes.add_edge(u_of_edge=parent, v_of_edge=i + 1, label=arc_label)
    return nodes


def parse_features(nodes, node):
    suffix_feats = "|".join([x for x in node.features["old"].split("|") if 'suf' in x])
    noun_feats = "|".join([x for x in node.features["old"].split("|") if 'suf' not in x])
    clean_suffix_feats = "|".join([x.replace("suf_", "") for x in node.features["old"].split("|") if 'suf' in x])
    if 'suf_' in node.features["old"] and node.pos["old"] == 'NN':
        node.lemma["new"] = '_'
        node.pos["new"] = 'NOUN'
        node.features["new"] = f"Definite=Def|{noun_feats}|xx_UD=Seg"

        # nodes.add_node(,
        #                word=word,
        #                pos={"new": "", "old": pos},
        #                lemma={"new": "", "old": lemma},
        #                features={"new": "", "old": features},
        #                arc_label={"new": "", "old": arc_label},
        #                parent={"new": "", "old": parent},
        #                )




def convert_nodes(nodes: nx.DiGraph) -> nx.DiGraph:
    for node in nodes:
        pass

    return nodes





def segment_df(self):
    output_df = pd.DataFrame(
        columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC',
                 ])
    print("segmenting sentence...")
    for i, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
        suffix_feats = "|".join([x for x in row['FEATS'].split("|") if 'suf' in x])
        noun_feats = "|".join([x for x in row['FEATS'].split("|") if 'suf' not in x])
        clean_suffix_feats = "|".join([x.replace("suf_", "") for x in row['FEATS'].split("|") if 'suf' in x])
        if 'suf_' in row['FEATS'] and row['UPOS'] == 'NN':
            output_df = output_df.append({'ID': row['ID'], 'FORM': row['LEMMA'] + '_', 'LEMMA': row['LEMMA'],
                                          'UPOS': 'NOUN', 'XPOS': 'NOUN',
                                          'FEATS': 'Definite=Def|' + noun_feats + '|xx_UD=Seg',
                                          'HEAD': row['HEAD'], 'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'],
                                          'MISC': row['MISC']}, ignore_index=True)

            output_df = output_df.append({'ID': 0, 'FORM': '_של_', 'LEMMA': 'של', 'UPOS': 'ADP',
                                          'XPOS': 'ADP', 'FEATS': '_' + '|xx_UD_REL=case:gen',
                                          'HEAD': int(row['ID']) + 2,
                                          'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'], 'MISC': row['MISC']}, ignore_index=True)

            output_df = output_df.append(
                {'ID': 0, 'FORM': self.pronouns[suffix_feats], 'LEMMA': 'הוא', 'UPOS': 'PRON',
                 'XPOS': 'PRON',
                 'FEATS': "Case=Gen|" + clean_suffix_feats + "|PronType=Prs" + '|xx_UD_REL=nmod:poss',
                 'HEAD': int(row['ID']) + 2, 'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'],
                 'MISC': row['MISC']}, ignore_index=True)

        # #TODO: check why this is here
        elif row['XPOS'] == 'S_PRN':
            if output_df.loc[i-1]['XPOS'] == 'AT':
                output_df.at[i - 1, 'FEATS'] = 'Case=Acc'
            elif output_df.loc[i - 1]['LEMMA'] == 'של':
                output_df.at[i - 1, 'FEATS'] = 'Case=Gen'
            elif output_df.loc[i - 1]['XPOS'] == 'IN':
                output_df.at[i - 1, 'FEATS'] = ''
            output_df = output_df.append(
            {'ID': row['ID'], 'FORM': '_' + row['LEMMA'], 'LEMMA': row['LEMMA'], 'UPOS': 'PRON',
             'XPOS': 'PRON', 'FEATS': row['FEATS'] + '|PronType=Prs', 'HEAD': row['HEAD'],
             'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'], 'MISC': row['MISC']}, ignore_index=True)


            # if output_df.loc[i - 1]['FEATS']:
            #     print("feats")
            #     prev_feats = output_df.loc[i - 1]['FEATS'] + '|'
            # else:
            #     print(self.df.loc[i-1])
            #     print(self.df.loc[i])
            #     print("no feats")
            #     prev_feats = output_df.loc[i - 1]['FEATS']
            # if prev_feats == '_|':
            #     prev_feats = ''
            # output_df = output_df.append(
            # {'ID': row['ID'], 'FORM': '_' + row['LEMMA'], 'LEMMA': row['LEMMA'], 'UPOS': 'PRON',
            #  'XPOS': 'PRON', 'FEATS': prev_feats + 'PronType=Prs', 'HEAD': row['HEAD'],
            #  'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'], 'MISC': row['MISC']}, ignore_index=True)


            # output_df.at[i - 1, 'XPOS'] = 'ADP'
            # output_df.at[i - 1, 'FORM'] += '_'
            # output_df.at[i - 1, 'FEATS'] = 'Case=Gen' # needs recheck - S_PRN seems mostly dative

        elif row['XPOS'] == 'DTT' or row['XPOS'] == 'DT':
            if 'suf_' in row['FEATS']:
                output_df = output_df.append(
                    {'ID': row['ID'], 'FORM': row['FORM'], 'LEMMA': row['LEMMA'], 'UPOS': 'NOUN',
                     'XPOS': 'NOUN', 'FEATS': row['FEATS'], 'HEAD': row['HEAD'],
                     'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'], 'MISC': row['MISC']}, ignore_index=True)

                output_df = output_df.append(
                    {'ID': 0, 'FORM': "_" + self.pronouns[suffix_feats], 'LEMMA': 'הוא', 'UPOS': 'PRON',
                     'XPOS': 'PRON',
                     'FEATS': "Case=Gen|" + clean_suffix_feats + "|PronType=Prs" + '|xx_UD_REL=nmod:poss',
                     'HEAD': int(row['ID']) + 1, 'DEPREL': row['DEPREL'], 'DEPS': row['DEPS'],
                     'MISC': row['MISC']}, ignore_index=True)

            else:
                output_df = output_df.append(row, ignore_index=True)
        else:
            output_df = output_df.append(row, ignore_index=True)
    return output_df


def apply_conversions(self, feats=None, simple_pos=None, complex_pos_conversions=None):

    def change_previous_row(row):
        """this method doesn't work yet. reise when the
        rest of the segmentation is fixed"""
        feats = row['FEATS']
        xpos = row['XPOS']
        upos = xpos
        prev = row.name - 1
        if xpos == 'PRON' and feats == 'PronType=Prs':
            if prev > 0:
                if self.df.at[prev, 'XPOS'] == 'ADP':
                    try:
                        prev_feats = self.df.at[prev, 'FEATS']
                    except:
                        print(prev)
                    self.segmented_sentence.at[prev, 'XPOS'] = 'ADP'
                    self.segmented_sentence.at[prev, 'FEATS'] = 'Case=Gen'
                    upos = 'PRON'
                    feats += '|PronType=Prs'
                else:
                    print(self.df.at[prev, 'XPOS'])
            else:
                print("here: name", row.name)
        else:
            print(xpos)
        return pd.Series([upos, feats])

    # self.segmented_sentence[['UPOS', 'FEATS']] = self.segmented_sentence.apply(change_previous_row, axis=1)

    def simple_features_conversion(column, conversions):
        for old, new in conversions.items():
            column = column.replace(old, new)

        return column

    def pos_conversion(column, conversions):
        if column in conversions:
            return conversions[column]
        else:
            return column

    def pos_convert_entire_line(row, conversions):
        xpos = row['XPOS']
        form = row['FORM']
        feats = row['FEATS']
        #     if xpos in conversions:
        if xpos in conversions:
            if 'concat' in xpos:
                if xpos['concat'] == 'before':
                    form = '_' + form
                elif xpos['concat'] == 'after':
                    form += '_'
                else:
                    form = '_' + form + '_'
            upos = conversions[xpos]['pos']
            if conversions[xpos]['feats'] == 'feats':
                feats = row['FEATS']
            elif conversions[xpos]['feats']['old'] == '_':
                feats = conversions[xpos]['feats']['new']
            elif conversions[xpos]['feats']['old'] == 'feats+':
                if len(row['FEATS']) > 2:
                    feats = row['FEATS'] + conversions[xpos]['feats']['new']
                else:
                    feats = conversions[xpos]['feats']['new'][1:]
            elif conversions[xpos]['feats']['old'] == '+feats':
                feats = conversions[xpos]['feats']['new'] + row['FEATS']
            elif conversions[xpos]['feats']['old'] == '+feats+':
                feats = conversions[xpos]['feats']['new'][0] + row['FEATS'] + conversions[xpos]['feats']['new'][1]
            return pd.Series([upos, form, feats])
        else:
            return pd.Series([row['UPOS'], row['FORM'], row['FEATS']])

    if feats:
        self.segmented_sentence.loc[:, 'FEATS'] = self.segmented_sentence['FEATS'].apply(
            lambda x: simple_features_conversion(x, feats))

    if simple_pos:
        self.segmented_sentence.loc[:, 'UPOS'] = self.segmented_sentence['UPOS'].apply(
            lambda x: pos_conversion(x, simple_pos))

    if complex_pos_conversions:
        self.segmented_sentence[['UPOS', 'FORM', 'FEATS']] = self.segmented_sentence.apply(
            lambda x: pos_convert_entire_line(x, complex_pos_conversions), axis=1)


if __name__ == '__main__':
    from yap_api import YapApi
    import matplotlib.pyplot as plt

    text = "גנן גידל דגן בגנו"
    ip = '127.0.0.1:8000'
    yap = YapApi()
    parsed_text = yap.run(text, ip)
    _dep_tree, _md_lattice, _ma_lattice, _segmented_text, _lemmas = yap.parse_sentence(text, ip)
    # print(len(_dep_tree), type(_dep_tree))
    # print(_md_lattice.columns)
    # print(len(_md_lattice), type(_md_lattice))
    print(_dep_tree.T)
    print(_dep_tree)
    tree = convert_sentence_to_graph(_dep_tree)
    # converter.apply_conversions(feats=basic_features, simple_pos=basic_pos, complex_pos_conversions=entire_line_pos_conversion)


    pos = nx.spring_layout(tree)
    nx.draw_networkx_nodes(tree, pos, cmap=plt.get_cmap('jet'), node_size = 500)
    nx.draw_networkx_labels(tree, pos)
    nx.draw_networkx_edges(tree, pos, edgelist=tree.edges, arrows=True)
    plt.show()

