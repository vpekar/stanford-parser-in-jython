"""A Jython interface to the Stanford parser (v.3.5.0). Includes various
utilities to manipulate parsed sentences:
* parse text containing XML tags,
* obtain probabilities for different analyses,
* extract dependency relations,
* extract subtrees,
* find the shortest path between two nodes,
* print the parse in various formats.

See examples after the if __name__ == "__main__" hooks.


INSTALLATION:

    1. Download the parser from http://nlp.stanford.edu/downloads/lex-parser.shtml
    2. Unpack into a local dir, put the path to stanford-parser.jar into the
    classpath for jython
    3. Put the full path to englishPCFG.ser.gz as parser_file arg to
    StanfordParser (searched in the local directory by default)

USAGE:

    Initialize a parser:

        parser = StanfordParser('englishPCFG.ser.gz')

    To keep XML tags provided in the input text:

        sentence = parser.parse('This is a <tag>test</tag>.')

    To strip all XML before parsing:

        sentence = parser.parse_xml('This is a <b>test</b>.')

    To print the sentence as a table (one word per line):

        sentence.print_table()

    To print the sentence as a parse tree:

        sentence.print_tree()

On input, the script accepts unicode or utf8 or latin1.

On output, the script produces unicode.
"""

__author__ = "Viktor Pekar <v.pekar@gmail.com>"
__version__ = "0.2"


import sys
import re
import os
import string
import math

try:
    assert 'java' in sys.platform
except AssertionError:
    raise Exception("The script should be run from Jython!")

from java.util import *
from edu.stanford.nlp.trees import PennTreebankLanguagePack, TreePrint
from edu.stanford.nlp.parser.lexparser import LexicalizedParser
from edu.stanford.nlp.process import Morphology, PTBTokenizer, WordTokenFactory
from edu.stanford.nlp.parser.lexparser import Options
from edu.stanford.nlp.ling import Sentence, WordTag
from java.io import StringReader


def stanford2tt(sentence):
    """Given a Sentence object, return TreeTagger-style
    tuples (word, tag, lemma).
    """

    for idx in sorted(sentence.word):

        word = sentence.word.get(idx, '')

        if word.startswith('<'):
            tag, lemma = 'XML', word
        else:
            tag = sentence.tag.get(idx, '')
            lemma = sentence.lemma.get(idx, '')

        # correcting: TO -> IN
        if word == 'to' and tag == 'TO':
            tag = 'IN'

        yield (word, tag, lemma)


class PySentence:
    """An interface to the grammaticalStructure object of SP
    """

    def __init__(self, parser, parse, xmltags={}):
        """Create a PySentence object from parse.
        @param gsf: a grammaticalStructureFactory object
        @param parse: a parse of the sentence
        @param xmltags: index of the previous text token =>
            list of intervening xmltags
        """
        self.gs = parser.gsf.newGrammaticalStructure(parse)
        self.parse = parse

        self.node = {}
        self.word = {}
        self.tag = {}
        self.lemma = {}
        self.dep = {}
        self.rel = {}
        self.children = {}

        self.lemmer = parser.lemmer
        self.xmltags = xmltags

        self.populate_indices()

    def get_lemma(self, word, tag):
        lemma = self.lemmer.lemmatize(WordTag(word, tag)).lemma()
        return lemma.decode('latin1')

    def get_pos_tag(self, node):
        parent = node.parent()
        tag = 'Z' if parent == None else parent.value()
        return tag.decode('latin1')

    def get_word(self, node_i):
        word = node_i.value().decode('latin1')

        # correct the appearance of parentheses
        if word == '-RRB-':
            word = u'('
        elif word == '-LRB-':
            word = u')'

        return word

    def populate_indices(self):

        # insert the tags before the text, if any are present before the text
        self.add_xml_tags_to_word_index(idx=0)

        # dependency indices
        for td in self.gs.typedDependenciesCCprocessed(True):
            dep_idx = td.dep().index()
            p_idx = td.gov().index()
            self.rel[dep_idx] = td.reln().getShortName()
            self.dep[dep_idx] = p_idx
            self.children[p_idx] = self.children.get(p_idx, [])
            self.children[p_idx].append(dep_idx)

        # word, pos tag and lemma indices
        for node_i in self.gs.root():

            if node_i.headTagNode() != None:
                continue

            idx = node_i.index()
            word = self.get_word(node_i)
            if word == "ROOT":
                break
            tag = self.get_pos_tag(node_i)

            self.node[idx] = node_i
            self.word[idx] = word
            self.tag[idx] = tag
            self.lemma[idx] = self.get_lemma(word, tag)

            # if the word is unattached
            if word in string.punctuation or not self.dep.get(idx):
                self.dep[idx] = 0
                self.rel[idx] = 'punct'

            # insert xml tags, if any
            self.add_xml_tags_to_word_index(idx)

    def add_xml_tags_to_word_index(self, idx):
        """@param idx: the id of the previous word
        """
        tags_at_idx = self.xmltags.get(idx)
        if tags_at_idx:
            num_tags = len(tags_at_idx)
            for tag_i in xrange(num_tags):
                tag_idx = (tag_i + 1) / float(num_tags + 1)
                tag_name = tags_at_idx[tag_i].decode('latin1')
                self.word[idx + tag_idx] = tag_name

    def get_head(self, node):
        """Return a tuple with the head of the dependency for a node and the
        relation label.
        """
        idx = node.index()
        dep_idx = self.dep.get(idx)
        if not dep_idx:
            return None, None
        return self.node.get(dep_idx), self.rel.get(idx)

    def get_children(self, node):
        """Yield tuples each with a child of the dependency
        and the relation label
        """
        for idx in self.children.get(node.index(), []):
            yield self.node[idx], self.rel[idx]

    def get_descendants(self, start_idx):
        """Return all descendants of a node, including the node itself
        """
        def traverse(idx):
            global descendants
            for idx_i in self.children.get(idx, []):
                descendants.append(idx_i)
                traverse(idx_i)
        global descendants
        descendants = [start_idx]
        traverse(start_idx)
        return descendants

    def prune(self, idx):
        """Given an index, remove all the words dependent on the word with that
        index, including the word itself.
        """
        for idx_i in self.get_descendants(idx):
            self.delete_node(idx_i)

    def delete_node(self, idx):
        del self.node[idx], self.word[idx], self.tag[idx], self.lemma[idx], \
                self.rel[idx], self.dep[idx]
        if idx in self.children:
            del self.children[idx]

    def get_plain_text(self):
        """Output plain-text sentence.
        """
        text = ' '.join([self.word[x] for x in sorted(self.node)])
        # remove spaces in front of commas, etc
        for i in ',.:;!?':
            text = text.replace(' ' + i, i)
        return text

    def get_least_common_node(self, node_i_idx, node_j_idx):
        """Return a node that is least common for two given nodes,
        as well as the shortest path between the two nodes
        @param node_i_idx: index of node 1
        @param node_j_idx: index of node 2
        """

        common_node = None
        shortest_path = []
        path1 = self.path2root(node_i_idx)
        path2 = self.path2root(node_j_idx)

        for idx_i in path1:
            if common_node != None:
                break
            for idx_j in path2:
                if idx_i == idx_j:
                    common_node = idx_i
                    break

        if common_node != None:
            for idx_i in path1:
                shortest_path.append(idx_i)
                if idx_i == common_node:
                    break
            for idx_i in path2:
                if idx_i == common_node:
                    break
                shortest_path.append(idx_i)

        return common_node, shortest_path

    def path2root(self, idx):
        """The path to the root from a node.
        @param idx: the index of the node
        """
        path = [idx]

        if idx != 0:
            while True:
                parent = self.dep.get(idx)
                if not parent:
                    break
                path.append(parent)
                idx = parent

        return path

    def print_table(self):
        """Print the parse as a table, FDG-style, to STDOUT
        """
        def get_index(id_str):
            return '-' if '.' in id_str else id_str

        for idx in sorted(self.word):
            line = '\t'.join([
                    get_index(unicode(idx)),
                    self.word.get(idx, ''),
                    self.lemma.get(idx, ''),
                    self.tag.get(idx, ''),
                    self.rel.get(idx, ''),
                    unicode(self.dep.get(idx, '')),
                ])
            print line.encode('latin1')

    def print_tree(self, mode='penn'):
        """Prints the parse.
        @param mode: penn/typedDependenciesCollapsed/etc
        """
        tree_print = TreePrint(mode)
        tree_print.printTree(self.parse)


class StanfordParser:

    TAG = re.compile(r'<[^>]+>')

    def __init__(self, parser_file,
            parser_options=['-maxLength', '80', '-retainTmpSubcategories']):

        """@param parser_file: path to the serialised parser model
            (e.g. englishPCFG.ser.gz)
        @param parser_options: options
        """

        assert os.path.exists(parser_file)
        options = Options()
        options.setOptions(parser_options)
        self.lp = LexicalizedParser.getParserFromFile(parser_file, options)
        tlp = PennTreebankLanguagePack()
        self.gsf = tlp.grammaticalStructureFactory()
        self.lemmer = Morphology()
        self.word_token_factory = WordTokenFactory()
        self.parser_query = None

    def get_most_probable_parses(self, text, kbest=2):
        """Yield kbest parses of a sentence along with their probabilities.
        """
        if not self.parser_query:
            self.parser_query = self.lp.parserQuery()

        response = self.parser_query.parse(self.tokenize(text))

        if not response:
            raise Exception("The sentence cannot be parsed: %s" % text)

        for candidate_tree in self.parser_query.getKBestPCFGParses(kbest):
            py_sentence = PySentence(self, candidate_tree.object())
            prob = math.e ** candidate_tree.score()
            yield py_sentence, prob

    def parse(self, sentence):
        """Strips XML tags first.
        @param s: the sentence to be parsed, as a string
        @return: a Sentence object
        """
        sentence = self.TAG.sub('', sentence)
        tokens = [unicode(x) for x in self.tokenize(sentence)]
        parse = self.lp.apply(Sentence.toWordList(tokens))
        return PySentence(self, parse)

    def tokenize(self, text):
        reader = StringReader(text)
        tokeniser = PTBTokenizer(reader, self.word_token_factory, None)
        tokens = tokeniser.tokenize()
        return tokens

    def parse_xml(self, text):
        """Tokenise the XML text, remember XML positions, and then parse it.
        """

        # build a plain-text token list and remember tag positions
        xml_tags = {}
        sent = []

        for token in self.tokenize(text):
            token = unicode(token).replace(u'\xa0', ' ')

            if token.startswith('<'):
                cur_size = len(sent)
                xml_tags[cur_size] = xml_tags.get(cur_size, [])
                xml_tags[cur_size].append(token)
            else:
                sent.append(token)

        # parse
        parse = self.lp.apply(Sentence.toWordList(sent))

        return PySentence(self, parse, xml_tags)


def parse_xml_example(sp):
    print 'Parsing XML text'
    text = 'The quick brown <tag attr="term">fox<!-- this is a comment --></tag> jumped over the lazy dog.'
    print 'IN:', text
    sentence = sp.parse_xml(text)
    print 'OUT:'
    sentence.print_table()
    print '-' * 80


def parse_probabilities_example(sp):
    print 'Parse probabilities\n'
    text = 'I saw a man with a telescope.'
    print 'IN:', text
    for sentence, prob in sp.get_most_probable_parses(text, kbest=2):
        print 'Probability:', prob
        print 'Tree:'
        sentence.print_table()
        print '-' * 50
    print '-' * 80


def subtrees_example(sp):
    print 'Subtrees:'
    text = 'I saw a man with a telescope.'
    sentence = sp.parse(text)
    for subtree in sentence.parse.subTrees():
        print subtree
        print '-' * 50
    print '-' * 80


def get_dependencies_example(sp):
    print 'Dependencies:'
    text = 'I saw a man with a telescope.'
    tmpl = 'Head: %s (%d); dependent: %s (%d); relation: %s'
    sentence = sp.parse(text)
    for td in sentence.gs.allTypedDependencies():
        gov = td.gov()
        gov_idx = gov.index()
        dep = td.dep()
        dep_idx = dep.index()
        rel = td.reln()
        print tmpl % (gov.value(), gov_idx, dep.value(), dep_idx, rel)
    print '-' * 80


def get_common_path_example(sp):
    tmpl = 'Least common node for "%s" and "%s": "%s"'
    print 'Common path:'
    text = 'The quick brown fox jumped over a lazy dog.'
    print 'Text:', text
    i = 4
    j = 9
    sentence = sp.parse(text)
    lcn, shortest_path = sentence.get_least_common_node(i, j)
    print tmpl % (sentence.word[i], sentence.word[j], sentence.word[lcn])
    path = ' '.join([sentence.word[x] for x in sorted(shortest_path)])
    print 'Path: %s' % path


if __name__ == '__main__':

    # full path to parser file, e.g. englishPCFG.ser.gz
    parser_file = sys.argv[1]
    sp = StanfordParser(parser_file)

    parse_xml_example(sp)
    parse_probabilities_example(sp)
    subtrees_example(sp)
    get_dependencies_example(sp)
    get_common_path_example(sp)
