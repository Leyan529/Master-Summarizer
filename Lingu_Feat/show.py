from spacy import displacy
from collections import Counter
import en_core_web_sm
import networkx as nx

import spacy
from nltk import Tree
from spacy import displacy



nlp = en_core_web_sm.load()


def find_shortest_path(doc, source, target):
    """Load spacy's dependency tree into a networkx graph"""
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    return nx.shortest_path(graph, source=source, target=target), \
            nx.shortest_path_length(graph, source=source, target=target)

def show_tree(doc):
    def to_nltk_tree(node):
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
        else:
            return node.orth_
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


def show_displacy(doc):
    displacy.render(doc, style="dep")