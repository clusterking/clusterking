#!/usr/bin/env python3

# std
import unittest

# ours
import clusterking.util.metadata as metadata


class TestNestedDict(unittest.TestCase):
    def test_nested_dict(self):
        nd = metadata.nested_dict()
        nd[1][2][3] = None
        self.assertEqual(nd, {1: {2: {3: None}}})

    def test_turn_into_nested_dict(self):
        a = {1: {2: 3, 3: {4: 5}}}
        a_nd = metadata.turn_into_nested_dict(a)
        self.assertEqual(a_nd[1][3]["nexist"], {})
        self.assertEqual(a_nd["nexist"], {})


class TestMetaData(unittest.TestCase):
    def test_save_load(self):
        gi = metadata.get_git_info()
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
            [[1, 2, 3], [4, 5], "xyz"],
        ]
        for case in cases:
            self.assertEqual(metadata.failsafe_serialize(case), case)
        self.assertEqual(metadata.failsafe_serialize(cases), cases)


class TestGetVersion(unittest.TestCase):
    def test_get_version(self):
        version = metadata.get_version()
        # Version has form int.int[.int]
        ints = version.split(".")
        self.assertGreaterEqual(len(ints), 2)
        self.assertLessEqual(len(ints), 3)
        for i in ints:
            self.assertTrue(i.isnumeric())


if __name__ == "__main__":
    unittest.main()
