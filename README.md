stanford-parser-in-jython
=========================

A Jython interface to the Stanford parser. Includes various utilities to manipulate
parsed sentences: 
* parsing text containing XML tags, 
* obtaining probabilities for different analyses,
* extracting dependency relations,
* extracting subtrees, 
* finding the shortest path between two nodes, 
* print the parse in various formats.

See examples after the if __ name __ == "__ main __" hooks.


INSTALLATION:

    1. Download the parser from http://nlp.stanford.edu/downloads/lex-parser.shtml
    2. Unpack into a local dir, put the path to stanford-parser.jar into the classpath for jython
    3. Put the path to englishPCFG.ser.gz as an arg to StanfordParser

USAGE: 

    Initialize a parser:

        parser = StanfordParser('englishPCFG.ser.gz')

    To keep XML tags provided in the input text:
    
        sentence = parser.parse('This is a test')
    
    To strip all XML before parsing:
    
        sentence = parser.parse_xml('This is a <b>test</b>.')
    
    To print the sentence as a table (one word per line):
    
        sentence.print_table()
    
    To print the sentence as a parse tree:
    
        sentence.print_tree()

On input, the script accepts unicode or utf8 or latin1.

On output, the script produces unicode.


