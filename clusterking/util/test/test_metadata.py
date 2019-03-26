#!/usr/bin/env python3

# std
import unittest

# ours
import clusterking.util.metadata as metadata


class TestNestedDict(unittest.TestCase):
    def test_nested_dict(self):
        nd = metadata.nested_dict()
        nd[1][2][3] = None
        self.assertEqual(
            nd,
            {
                1: {
                    2: {
                        3: None
                    }
                }
            }
        )


class TestMetaData(unittest.TestCase):

    def test_save_load(self):
        gi = metadata.git_info()
        gi_saved = metadata.save_git_info()
        self.assertEqual(gi, gi_saved)
        gi_loaded = metadata.load_git_info()
        self.assertEqual(gi_saved, gi_loaded)

    def test_serialize_identical(self):
        cases = [
            "test",
            3,
            3.123,
            {1: 2},
            [1, 2, 3],
            [{1: 3}, {3: 4}],
            [[1, 2, 3], [4, 5], "xyz"]
        ]
        for case in cases:
            self.assertEqual(
                metadata.failsafe_serialize(case),
                case
            )
        self.assertEqual(
            metadata.failsafe_serialize(cases),
            cases
        )


if __name__ == "__main__":
    unittest.main()
