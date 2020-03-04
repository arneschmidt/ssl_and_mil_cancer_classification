import argparse
from src.data import generate_data
from src.model import ClassficationModel

def main(args):
    print("Create data generators..")
    train_data, val_data, test_data = generate_data(args.data_dir, args.data_split_dir, args.batch_size)

    print("Load classification model")
    classification_model = ClassficationModel(args.batch_size, args.load_model)

    if args.mode == "train":
        print("Train")
        classification_model.train(train_data, val_data, args.save_model)
    elif args.mode == "test":
        print("Test")
        classification_model.test(test_data)
    elif args.mode == "predict":
        print("Predict")
        classification_model.predict(test_data, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Classification")
    parser.add_argument("--mode", "-m", type=str, default="predict",
                        help="Choose mode: train or test")
    parser.add_argument("--data_dir", "-d", type=str, default="../data/",
                        help="Dataset directory.")
    parser.add_argument("--data_split_dir", "-f", type=str, default="../data_split/",
                        help="txt file with all filenames and paths")
    parser.add_argument("--save_model", "-s", type=str, default="../models/mobilenet.h5",
                        help="Path to save model.")
    parser.add_argument("--load_model", "-l", type=str, default="../models/mobilenet.h5",
                        help="Path to .h5 model to load for testing or retraining. Set to 'None' for new model.")
    parser.add_argument("--output_dir", "-o", type=str, default="../output/",
                        help="Path to .h5 model to load for testing or retraining. Set to 'None' for new model.")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                        help="Batch size")
    args = parser.parse_args()
    main(args)