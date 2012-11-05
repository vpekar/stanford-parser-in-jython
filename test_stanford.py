"""To run the tests, put englishPCFG.ser.gz into the working directory.

Created on 4 Nov 2012

@author: viktor
"""

import unittest
from stanford import StanfordParser, PySentence


class TestPySentence(unittest.TestCase):

    def setUp(self):
        input = "The quick brown fox jumped over a lazy dog."
        sp = StanfordParser('englishPCFG.ser.gz')
        self.sentence = sp.parse_xml(input)
        
    def test_get_least_common_node(self):        
        lcn, shortest_path = self.sentence.get_least_common_node(4, 9)
        actual_lcn = self.sentence.word[lcn]
        actual_path = ' '.join([self.sentence.word[x] for x in sorted(shortest_path)])
        expected_lcn = 'jumped'
        expected_path = 'fox jumped over dog'
        msg = "Expected %s != actual %s" % (expected_lcn, actual_lcn)
        self.assertTrue(expected_lcn == actual_lcn, msg)
        msg = "Expected %s != actual %s" % (expected_path, actual_path)
        self.assertTrue(expected_path == actual_path, msg)
        

class TestStanfordParser(unittest.TestCase):

    def setUp(self):
        self.sp = StanfordParser('englishPCFG.ser.gz')

    def test_get_most_probable_parses_check_types(self):
        input = 'I saw a man with a telescope.'
        expected_type1, expected_type2 = type(PySentence), float
        for s, prob in self.sp.get_most_probable_parses(input, kbest=2):
            actual_type1, actual_type2 = type(s.__class__), type(prob)
            msg = "Expected %s != actual %s" % (expected_type1, actual_type1)
            self.assertTrue(actual_type1 == expected_type1, msg)
            msg = "Expected %s != actual %s" % (expected_type2, actual_type2)
            self.assertTrue(actual_type2 == expected_type2, msg)

    def test_get_most_probable_parses_check_nonzero(self):
        input = 'I saw a man with a telescope.'
        expected = 2
        parses = [x for x in self.sp.get_most_probable_parses(input, kbest=expected)]
        actual = len(parses)
        msg = "Expected %d != actual %d" % (expected, actual)
        self.assertTrue(expected == actual, msg)
        
    def test_parse_xml(self):
        input = 'This <a>is</a> a test<!-- b -->.'
        expected = ['DT', 'VBZ', 'DT', 'NN', '.']
        sentence = self.sp.parse_xml(input)
        actual = [v for k, v in sorted(sentence.tag.items())]
        msg = "Expected %s != actual %s" % (expected, actual)
        self.assertTrue(expected == actual, msg)
        
    def test_tokenise(self):
        input = 'This is a test.'
        expected = ['This', 'is', 'a', 'test', '.']
        actual = [unicode(x) for x in self.sp.tokenize(input)]
        msg = "Expected %s != actual %s" % (expected, actual)
        self.assertTrue(expected == actual, msg)


if __name__ == "__main__":
    unittest.main()