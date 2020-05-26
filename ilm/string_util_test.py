import unittest

from ilm.string_util import *

_DOC = """
After Iowa's first-in-the-nation caucuses melted down earlier this week, all eyes are on the New Hampshire primary to give some clarity to the crowded and chaotic Democratic presidential race.
Voters in the New England state will head to the polls on February 11 to "pick" the Democrat they want to see nominated. The contest is the second stop after the all-important Iowa caucuses, which normally help cut down the field of candidates.
But issues in the Midwestern state have left the race more open-ended than usual. Historically, the two or three candidates who come out on top in Iowa see increased momentum for the rest of the early-voting season. Now, the muddled results have left the primary race in slight disarray.
"""

_VERSE = """
There once was a man from Peru,
Who dreamed he was eating his shoe.

He woke with a fright
In the middle of the night
To find that his dream had come true!
"""


class TestStringUtil(unittest.TestCase):

  def test_verse_to_hierarchical_offsets(self):
    doc = _VERSE
    doc_offs = doc_to_hierarchical_offsets(doc, verse=True)
    doc_offs_relative = doc_to_hierarchical_offsets(doc, verse=True, relative=True)

    def coord_test(coord, expected_off, expected_len, expected_num_children=None, expected_string=None):
      n = doc_offs
      for c in coord:
        self.assertEqual(type(n), tuple)
        self.assertEqual(len(n), 3)
        self.assertEqual(type(n[0]), int)
        self.assertEqual(type(n[1]), int)
        self.assertEqual(type(n[2]), tuple)
        self.assertTrue(c < len(n[2]))
        n = n[2][c]

      c_off, c_len = n[:2]
      self.assertEqual(c_off, expected_off)
      self.assertEqual(c_len, expected_len)

      if expected_num_children is None:
        self.assertEqual(len(n), 2)
      else:
        c_num_children = len(n[2])
        self.assertEqual(c_num_children, expected_num_children)

      if expected_string is not None:
        self.assertEqual(doc[c_off:c_off+c_len], expected_string)

    coord_test((), 0, 157, 2)

    coord_test((0,), 1, 67, 2)
    coord_test((0, 0), 1, 31, 8)
    coord_test((0, 0, 0), 1, 5, None, 'There')
    coord_test((0, 0, 6), 27, 4, None, 'Peru')

    coord_test((1,), 70, 86, 3)
    coord_test((1, 1), 92, 26, 6, 'In the middle of the night')
    coord_test((1, 1, 5), 113, 5, None, 'night')

  def test_doc_to_hierarchical_offsets(self):
    doc = _DOC
    doc_offs = doc_to_hierarchical_offsets(doc)
    doc_offs_relative = doc_to_hierarchical_offsets(doc, relative=True)

    def coord_test(coord, expected_off, expected_len, expected_num_children=None, expected_string=None):
      n = doc_offs
      for c in coord:
        self.assertEqual(type(n), tuple)
        self.assertEqual(len(n), 3)
        self.assertEqual(type(n[0]), int)
        self.assertEqual(type(n[1]), int)
        self.assertEqual(type(n[2]), tuple)
        self.assertTrue(c < len(n[2]))
        n = n[2][c]

      c_off, c_len = n[:2]
      self.assertEqual(c_off, expected_off)
      self.assertEqual(c_len, expected_len)

      if expected_num_children is None:
        self.assertEqual(len(n), 2)
      else:
        c_num_children = len(n[2])
        self.assertEqual(c_num_children, expected_num_children)

      if expected_string is not None:
        self.assertEqual(doc[c_off:c_off+c_len], expected_string)

    coord_test((), 0, 727, 3)

    coord_test((0,), 1, 192, 1)
    coord_test((0, 0), 1, 192, 32)
    coord_test((0, 0, 0), 1, 5, None, 'After')
    coord_test((0, 0, 2), 11, 2, None, '\'s')

    coord_test((1,), 194, 244, 2)
    coord_test((1, 1), 315, 123, 22)
    coord_test((1, 1, 5), 341, 4, None, 'stop')
    coord_test((1, 0, 15), 267, 1, None, '\"')
    coord_test((1, 0, 16), 268, 4, None, 'pick')
    coord_test((1, 0, 17), 272, 1, None, '\"')

    coord_test((2,), 439, 287, 3)


if __name__ == '__main__':
  unittest.main()
