"""grow_set_h.symbols_of: the companies a draw passes over, from invented sets.
Run: .venv/bin/python -I -m unittest discover tests"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from grow_set_h import symbols_of  # noqa: E402


class SymbolsOf(unittest.TestCase):
    def test_the_companies_of_every_set_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for name, symbols in (("a", ["AAA", "AAA", "BBB"]), ("b", ["CCC"])):
                paths.append(os.path.join(tmp, name + ".jsonl"))
                with open(paths[-1], "w", encoding="utf-8") as fh:
                    fh.write("".join(json.dumps({"symbol": s, "role": "base"}) + "\n" for s in symbols) + "\n")
            self.assertEqual(symbols_of(paths), {"AAA", "BBB", "CCC"})
            self.assertEqual(symbols_of(paths[1:]), {"CCC"})
        self.assertEqual(symbols_of([]), set())


if __name__ == "__main__":
    unittest.main()
