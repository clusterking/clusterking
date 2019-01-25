#!/usr/bin/env python3

# std
import unittest

# ours
import bclustering.util.metadata as metadata


class TestMetaData(unittest.TestCase):

    def test_save_load(self):
        gi = metadata.git_info()
        gi_saved = metadata.save_git_info()
        self.assertEqual(gi, gi_saved)
        gi_loaded = metadata.load_git_info()
        self.assertEqual(gi_saved, gi_loaded)


if __name__ == "__main__":
    unittest.main()