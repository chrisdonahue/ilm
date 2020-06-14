import re
import string

re_newlines = re.compile(r'\n\n[\n]+')

def sanitize_verse(s):
  # Remove non-ascii
  s = s.encode('ascii', 'ignore').decode('ascii')

  # Remove whitespace
  s = s.strip()

  # Split into lines
  lines = s.splitlines()

  # Remove punctuation
  lines = [' '.join([w.strip(string.punctuation) for w in l.split()]) for l in lines]

  # Remove egregious capslock
  lines = [l.lower() if l == l.upper() else l for l in lines]

  # Capitalize first letter
  lines = [l if len(l) == 0 else l[0].upper() + l[1:] for l in lines]

  # Remove redundant newlines
  s = '\n'.join(lines)
  s = re_newlines.sub('\n\n', s)

  return s
