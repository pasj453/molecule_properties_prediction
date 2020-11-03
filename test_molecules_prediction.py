import unittest
from main import get_parser, check_argument_parsing
from vectorization import load_dataset, vectorizes_label, vectorizes_features


class TestArgumentParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = get_parser()

    def test_train(self):
        args_ok = ["train", "--fname", "fixtures/dataset.csv"]
        args_nok_1 = ["train", "--fname", "fixtures/dataset_not_exist.csv"]
        args_nok_2 = ["train"]

        args_ok = self.parser.parse_args(args_ok)
        check_argument_parsing(args_ok)
        for nok_args in (args_nok_1, args_nok_2):
            with self.assertRaises(FileNotFoundError):
                check_argument_parsing(
                    self.parser.parse_args(nok_args)
                )

    def test_predict(self):
        args_ok = ["predict", "--model", "fixtures/test_model.h5",
                   "--fname", "fixtures/dataset.csv",
                   "--smile", "HOH"]
        args_nok_1 = ["predict"]
        args_nok_2 = ["predict", "--model", "fixtures/model.h5"]
        args_nok_3 = ["predict", "--model", "fixtures/test_model.h5",
                      "--fname", "fixtures/dataset.csv"]

        args_ok = self.parser.parse_args(args_ok)
        check_argument_parsing(args_ok)

        for args_nok in (args_nok_1, args_nok_2, args_nok_3):
            with self.assertRaises(Exception):
                check_argument_parsing(self.parser.parse_args(args_nok))

    def test_evaluate(self):
        args_ok = ["evaluate", "--model", "fixtures/test_model.h5",
                   "--fname", "fixtures/dataset.csv"]
        args_nok_1 = ["evaluate"]
        args_nok_2 = ["evaluate", "--model", "fixtures/model.h5"]
        args_nok_3 = ["evaluate", "--model", "fixtures/test_model.h5",
                      "--fname", "fixtures/dataset_not_exist.csv"]

        args_ok = self.parser.parse_args(args_ok)
        check_argument_parsing(args_ok)
        for nok_args in (args_nok_1, args_nok_2, args_nok_3):
            with self.assertRaises(FileNotFoundError):
                check_argument_parsing(
                    self.parser.parse_args(nok_args)
                )


class TestVectorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_dataset("fixtures/fixtures_dataset.csv")

    def test_morgan_fingerpriting(self):
        y = vectorizes_label(self.df)
        self.assertEqual(y.shape, (49, 1))

        x1 = vectorizes_features(self.df, **{"size": 2048})
        self.assertEqual(x1.shape, (49, 2048))

        x2 = vectorizes_features(self.df, **{"size": 1024})
        self.assertEqual(x2.shape, (49, 1024))


if __name__ == "__main__":
    unittest.main()
