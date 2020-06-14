# -*- coding: utf-8 -*- 

import unittest

from string_util import *

_VERSE_RAW = u"""

   
    
¿Dónde está?
We don't know...
This is crucial!
FOR ME TO SHOW
    
\t 
 
New stanza  is   \t here
A            \t\t\t     
\tHave no fear 45
I

\t
  
  \t\n
\n\n\n\t\t\t\t\t\t\n\n   \n\n\t\t

What is-life
life is strife
\n\n\t\t\t\t\n\n
"""


class TestStringUtil(unittest.TestCase):

  def test_sanitize_verse(self):
    verse_sanitized = sanitize_verse(_VERSE_RAW)
    self.assertEqual(len(verse_sanitized.splitlines()), 12)
    
    stanzas = verse_sanitized.split('\n\n')
    self.assertEqual(len(stanzas), 3)
    self.assertEqual(len(stanzas[0].splitlines()), 4)
    self.assertEqual(len(stanzas[1].splitlines()), 4)
    self.assertEqual(len(stanzas[2].splitlines()), 2)


if __name__ == '__main__':
  unittest.main()
