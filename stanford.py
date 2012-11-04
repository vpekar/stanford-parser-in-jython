"""A Jython interface to the Stanford parser (v.2.0.3). Includes various utilities
to manipulate parsed sentences: 
* parse text containing XML tags, 
* obtain probabilities for different analyses,
* extract dependency relations,
* extract subtrees, 
* find the shortest path between two nodes, 
* print the parse in various formats.

See examples after the if __name__ == "__main__" hooks.


INSTALLATION:

    1. Download the parser from http://nlp.stanford.edu/downloads/lex-parser.shtml
    2. Unpack into a local dir, put the path to stanford-parser.jar into the classpath for jython
    3. Put the full path to englishPCFG.ser.gz as parser_file arg to StanfordParser

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

__author__="Viktor Pekar <v.pekar@gmail.com>"
__version__="0.1"

import sys, re, os, string, math

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
    """Given a Sentence object, return TreeTagger-style tuples (word, tag, lemma).
    """
    for k in sorted(sentence.word):
        word = sentence.word.get(k, '')
        if word.startswith('<'):
            tag, lemma = 'XML', word
        else:
            tag = sentence.tag.get(k, '')
            lemma = sentence.lemma.get(k, '')
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
        @param xmltags: index of the previous text token => list of intervening xmltags
        """
        self.parse = parse
        self.gs = parser.gsf.newGrammaticalStructure(parse)
        self.lemmer = parser.lemmer
        self.xmltags = xmltags
        
        # create indices
        self.node = {}
        self.word = {}
        self.tag = {}
        self.lemma = {}
        self.dep = {}
        self.rel = {}
        self.children = {}
        
        # insert the tags before the text, if any are present before the text
        if 0 in self.xmltags:
            num_tags = len(self.xmltags[0])
            for idx in xrange(num_tags):
                tag_idx = (idx+1)/float(num_tags+1)
                self.word[tag_idx] = self.xmltags[0][idx].decode('latin1')
                
        # iterate over text tokens
        for i in self.gs.getNodes():
            if i.headTagNode() != None: continue
            idx = i.index()
            word = i.value().decode('latin1')
            
            # correction
            if word == '-RRB-': word = u'('
            elif word == '-LRB-': word = u')'
            
            parent = i.parent()
            tag = u'Z' if parent == None else parent.value().decode('latin1')
            lemma = self.lemmer.lemmatize(WordTag(word, tag)).lemma().decode('latin1')
            p = self.gs.getGovernor(i)
            if word in string.punctuation or p == None:
                p_idx = 0
                rel = 'punct'
            else:
                p_idx = p.index()
                rel = str(self.gs.getGrammaticalRelation(p_idx, idx))
            
            self.node[idx] = i
            self.word[idx] = word
            self.tag[idx] = tag
            self.lemma[idx] = lemma
            self.rel[idx] = rel
            self.dep[idx] = p_idx
            self.children[p_idx] = self.children.get(p_idx,[])
            self.children[p_idx].append( idx )
            
            # insert xml tags, if any
            if idx in self.xmltags:
                num_tags = len(self.xmltags[idx])
                for t_num in xrange(num_tags):
                    tag_idx = (t_num+1)/float(num_tags+1)
                    self.word[idx+tag_idx] = self.xmltags[idx][t_num].decode('latin1')

    def get_head(self, node):
        """Return a tuple with the head of the dependency for a node and the 
        relation label.
        """
        idx = node.index()
        dep_idx = self.dep.get(idx)
        if not dep_idx: return None, None
        return self.node.get(dep_idx), self.rel.get(idx)
    
    def get_children(self,node):
        """Yield tuples each with a child of the dependency 
        and the relation label
        """
        for i in self.children.get(node.index(), []):
            yield self.node[i], self.rel[i]
    
    def descendants(self,idx):
        """Return all descendants of a node, including the node itself
        """
        global descendants
        descendants = [idx]
        def traverse(idx):
            global descendants
            for i in self.children.get(idx, []):
                descendants.append(i)
                traverse(i)
        traverse(idx)
        return descendants
    
    def prune(self,idx):
        """Given an index, remove all the words dependent on the word with that index,
        including the word itself.
        """
        for i in self.descendants(idx):
            self.delete_node(i)
                
    def delete_node(self,i):
        del self.node[i], self.word[i], self.tag[i], self.lemma[i], self.rel[i], self.dep[i]
        if i in self.children:
            del self.children[i]

    def get_plain_text(self):
        """Output plain-text sentence.
        """
        text = ' '.join([self.word[i] for i in sorted(self.node)])
        # remove spaces in front of commas, etc
        for i in ',.:;!?':
            text = text.replace(' ' + i, i)
        return text

    def get_least_common_node(self,n,m):
        """Return a node that is least common for two given nodes, 
        as well as the shortest path between the two nodes
        @param n: index of node 1
        @param m: index of node 2
        """
       
        common_node = None
        shortest_path = []
        path1 = self.path2root(m)
        path2 = self.path2root(n)
        
        for i in path1:
            if common_node != None: break
            for j in path2:
                if i == j:
                    common_node = i
                    break
        
        if common_node != None:
            for i in path1:
                shortest_path.append(i)
                if i == common_node: break
            for i in path2:
                if i == common_node: break
                shortest_path.append(i)
        
        return common_node, shortest_path
    
    def path2root(self, i):
        """The path to the root from a node.
        @param i: the index of the node 
        """
        path = [i]
        if i != 0: 
            while 1:
                p = self.dep.get(i)
                if not p: break
                path.append(p)
                i = p
        return path
    
    def print_table(self):
        """Print the parse as a table, FDG-style, to STDOUT
        """
        def get_index(s):
            return '-' if '.' in s else s
        for i in sorted(self.word):
            line = '\t'.join([
                    get_index(unicode(i)),
                    self.word.get(i,''),
                    self.lemma.get(i,''),
                    self.tag.get(i,''),
                    self.rel.get(i,''),
                    unicode(self.dep.get(i,'')),
                ])
            print line.encode('latin1')
    
    def print_tree(self, mode='penn'):
        """Prints the parse.
        @param mode: penn/typedDependenciesCollapsed/etc
        """
        tp = TreePrint(mode)
        tp.printTree(self.parse)

class StanfordParser:
    
    TAG = re.compile(r'<[^>]+>')
    
    def __init__(self, parser_file, 
            parser_options=['-maxLength', '80', '-retainTmpSubcategories']):
        """@param parser_file: path to the serialised parser model (e.g. englishPCFG.ser.gz)
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
        response = self.parser_query.parse(sp.tokenize(text))
        if not response:
            raise Exception("The sentence was not accepted by the parser: %s" % text)
        for candidate_tree in self.parser_query.getKBestPCFGParses(kbest):
            s = PySentence(sp, candidate_tree.object())
            prob = math.e**candidate_tree.score()
            yield s, prob

    def parse(self, s):
        """Strips XML tags first.
        @param s: the sentence to be parsed, as a string
        @return: a Sentence object
        """
        # strip xml tags
        s = self.TAG.sub('', s)
        
        parse = self.lp.apply(s)
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
        for i in self.tokenize(text):
            token = unicode(i)
            if token.startswith('<'):
                cur_size = len(sent)
                xml_tags[cur_size] = xml_tags.get(cur_size,[])
                xml_tags[cur_size].append(token)
            else:
                sent.append(token)
                
        # parse
        parse = self.lp.apply(Sentence.toWordList(sent))
        
        return PySentence(self, parse, xml_tags)

def parse_xml_example(sp):
    print 'Parsing XML text'
    s = 'The quick brown <tag attr="term">fox<!-- this is a comment --></tag> jumped over the lazy dog.'
    print 'IN:', s
    sentence = sp.parse_xml(s)
    print 'OUT:'
    sentence.print_table()
    print '-'*80
    
def parse_probabilities_example(sp):
    print 'Parse probabilities\n'
    text = 'I saw a man with a telescope.'
    print 'IN:', text
    for s, prob in sp.get_most_probable_parses(text, kbest=2):
        print 'Probability:', prob
        print 'Tree:'
        s.print_table()
        print '-'*50
    print '-'*80
    
def subtrees_example(sp):
    print 'Subtrees:'
    text = 'I saw a man with a telescope.'
    sentence = sp.parse(text)
    for subtree in sentence.parse.subTrees():
        print subtree
        print '-'*50
    print '-'*80
    
def get_dependencies_example(sp):
    print 'Dependencies:'
    text = 'I saw a man with a telescope.'
    sentence = sp.parse(text)
    for td in sentence.gs.allTypedDependencies():
        gov = td.gov()
        gov_idx = gov.index()
        dep = td.dep()
        dep_idx = dep.index()
        rel = td.reln()
        print 'Head: %s (%d); dependent: %s (%d); relation: %s' % (gov.value(), gov_idx, dep.value(), dep_idx, rel)
    print '-'*80
    
def get_common_path_example(sp):
    print 'Common path:'
    text = 'The quick brown fox jumped over the lazy dog.'
    print 'Text:', text
    i = 4
    j = 9
    sentence = sp.parse(text)
    lcn, shortest_path = sentence.get_least_common_node(i, j)
    print 'Least common node for "%s" and "%s": "%s"' % (sentence.word[i], sentence.word[j], sentence.word[lcn])
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

    
