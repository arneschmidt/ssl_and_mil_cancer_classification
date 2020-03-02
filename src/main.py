import argparse
from src.data import generate_data
from src.model import ClassficationModel

def main(args):
    data = generate_data(args.data_dir, args.filenames, args.batch_size)

    classification_model = ClassficationModel(args.batch_size)
    classification_model.train(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Classification")
    parser.add_argument("--data_dir", "-d", type=str, default="../data/",
                        help="Dataset directory.")
    parser.add_argument("--filenames", "-f", type=str, default="../filenames.txt",
                        help="txt file with all filenames and paths")
    parser.add_argument("--save_model_path", "-s", type=str, default="./output",
                        help="txt file with all filenames and paths")
    parser.add_argument("--batch_size", "-b", type=int, default=12,
                        help="Batch size")
    args = parser.parse_args()
    main(args)