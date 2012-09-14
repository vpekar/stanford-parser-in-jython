"""A Jython interface to the Stanford parser. Includes various utilities to manipulate
parsed sentences: 
* parsing text containing XML tags, 
* obtaining probabilities for different analyses,
* extracting dependency relations,
* extracting subtrees, 
* finding the shortest path between two nodes, 
* print the parse in various formats.

See examples after the if __name__ == "__main__" hooks.


INSTALLATION:

    1. Download the parser from http://nlp.stanford.edu/downloads/lex-parser.shtml
    2. Unpack into a local dir, put the path to stanford-parser.jar in the -cp arg in jython.bat
    3. Put the path to englishPCFG.ser.gz as parser_file arg to StanfordParser

USAGE: 

1. Produce an FDG-style of a parse (a table as a list of words with tags):

        parser = StanfordParser()

    To keep XML tags provided in the input text:
    
        sentence = parser.parse('This is a test')
    
    To strip all XML before parsing:
    
        sentence = parser.parse_xml('This is a <b>test</b>.')
    
    To print the sentence as a table (one word per line):
    
        sentence.print_table()
    
    To print the sentence as a parse tree:
    
        sentence.print_tree()
    
2. Retrieve the 5 best parses with associated probabilities for the last-parsed sentence:

    parser = StanfordParser()
    sentence = parser.parse('This is a test')
    for candidate_tree in parser.lp.getKBestPCFGParses(5):
        print 'Prob:', math.e**candidate_tree.score()
        print 'Tree:'
        s = Sentence(parser.gsf, candidate_tree.object())
        s.print_table()

On input, the script accepts unicode or utf8 or latin1.
On output, the script produces unicode.
"""

__author__="Viktor Pekar <v.pekar@gmail.com>"
__version__="0.1"

import sys, re, string, math

try:
    assert 'java' in sys.platform
except AssertionError:
    raise Exception("The script should be run from Jython!")

from java.util import *
from edu.stanford.nlp.trees import PennTreebankLanguagePack, TreePrint
from edu.stanford.nlp.parser.lexparser import LexicalizedParser
from edu.stanford.nlp.process import Morphology, PTBTokenizer, WordTokenFactory
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


class Sentence:
    """An interface to the grammaticalStructure object of SP
    """
    
    def __init__(self, gsf, parse, xmltags={}):
        """Create a Sentence object from parse.
        @param gsf: a grammaticalStructureFactory object
        @param parse: a parse of the sentence
        @param xmltags: index of the previous text token => list of intervening xmltags
        """
        self.parse = parse
        self.gs = gsf.newGrammaticalStructure(parse)
        self.lemmer = Morphology()
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
            lemma = self.lemmer.lemmatize(self.lemmer.stem(word, tag)).lemma().decode('latin1')
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
        return (self.node.get(dep_idx), self.rel.get(idx))
    
    def get_children(self,node):
        """Yield tuples each with a child of the dependency 
        and the relation label
        """
        for i in self.children.get(node.index(), []):
            yield (self.node[i], self.rel[i])
    
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

    def least_common_node(self,n,m):
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
        for i in sorted(self.word):
            line = '\t'.join([
                    self.word.get(i,'')
                    self.lemma.get(i,'')
                    self.tag.get(i,'')
                    self.rel.get(i,'')
                    self.dep.get(i,'')
                ])
            print line.encode('utf8')
    
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
        self.lp = LexicalizedParser(parser_file)
        self.lp.setOptionFlags(parser_options)
        tlp = PennTreebankLanguagePack()
        self.gsf = tlp.grammaticalStructureFactory()
        self.wtf = WordTokenFactory()

    def parse(self, s):
        """Strips XML tags first.
        @param s: the sentence to be parsed, as a string
        @return: a Sentence object
        """
        # strip xml tags
        s = self.TAG.sub('', s)
        
        parse = self.lp.apply(s)
        return Sentence(self.gsf, parse)
        
    def parse_xml(self,s):
        """Tokenise the XML text, remember XML positions, and then parse it.
        """
        
        # tokenise the text
        r = StringReader(s)
        tokeniser = PTBTokenizer(r, False, self.wtf)
        alist = tokeniser.tokenize()
        
        # build a plain-text token list and remember tag positions
        tags = {}
        sent = []
        for i in alist:
            token = str(i)
            if token.startswith('<'):
                cur_size = len(sent)
                tags[cur_size] = tags.get(cur_size,[])
                tags[cur_size].append(token)
            else:
                sent.append(token)
                
        # parse
        parse = self.lp.apply(Arrays.asList(sent))
        
        return Sentence(self.gsf, parse, tags)


if __name__ == '__main__':
        
    sp = StanfordParser(r'C:\soft\stanford\stanford-parser-2008-10-26\englishPCFG.ser.gz')
    
    print 'Parsing XML text\n'
    s = 'This is an <tag attr="term">example<!-- this is a comment --></tag>.'
    print 'IN:', s
    sentence = sp.parse_xml(s)
    print 'OUT:'
    sentence.print_table()
    print '-'*80
    
    print 'Output formats\n'
    s = 'This is an example sentence.'
    print 'IN:', s
    sentence = sp.parse_xml(s)
    print 'TABLE:'
    sentence.print_table()
    print '\nTREE:'
    sentence.print_tree()
    print '\nTT FORMAT:'
    for i in stanford2tt(sentence):
        print i
    print '-'*80
    
    print 'Parse probabilities\n'
    s = 'I saw a man with a telescope.'
    print 'IN:', s
    for candidate_tree in sp.lp.getKBestPCFGParses(1):
        print 'Probability:', math.e**candidate_tree.score()
        print 'Tree:'
        s = Sentence(sp.gsf, candidate_tree.object())
        s.print_table()
        print '-'*50
    print '-'*80
    
    """
    print
    print 'Subtrees:\n'
    for subtree in sentence.parse.subTrees():
        print subtree
        print '-'*50
    print '-'*80
    """
    
    print 'Dependencies\n'
    for td in sentence.gs.allTypedDependencies():
        gov = td.gov()
        gov_idx = gov.index()
        dep = td.dep()
        dep_idx = dep.index()
        rel = td.reln()
        print 'Governing word:',gov.value()
        print 'Its index:',gov_idx
        print 'Dependency word:',dep.value()
        print 'Its index:',dep_idx
        print '-'*50
    print '-'*80
    
    """
    # paths between every pair of content words
    content = []
    for i in sentence.gs.getNodes():
        if i.headTagNode() != None: continue
        idx = i.index()
        word = i.value()
        tag = i.parent().value()
        if tag[0] in ['V','N','J','R']:
            content.append(i)
    for i in content:
        for j in content:
            if i == j: continue
            lcn, shortest_path = sentence.least_common_node(i.index(), j.index())
            print 'LCN: %s and %s: %s' % (i, j, lcn)
            print 'Path:', shortest_path
            print '-'*50
    """
    
