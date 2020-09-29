import argparse
import tensorflow as tf
from typing import Dict, Optional, Tuple
from src.data import generate_data
from src.model import ClassficationModel

def main(args):
    devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(devices[0], True)
    print("Create data generators..")
    train_data, val_data, test_data = generate_data(args.dataset_name, args.artifact_base_folder, args.data_dir,
                                                    args.data_split_dir, args.image_target_size, args.batch_size)

    print("Load classification model")
    classification_model = ClassficationModel(args.batch_size, args.load_model, args.model_architecture)

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
    parser.add_argument("--mode", "-m", type=str, default="train",
                        help="Choose mode: train or test")
    parser.add_argument("--model_architecture", "-ma", type=str, default="efficientnetb0",
                        help="Choose model architecture: mobilenetv2, efficientnetb0 or simple_cnn")
    parser.add_argument("--dataset_name", "-dn", type=str, default="sicapv2",
                        help="Dataset directory.")
    parser.add_argument("--data_dir", "-dd", type=str, default="/home/arne/datasets/SICAPv2/images",
                        help="Dataset directory.")
    parser.add_argument("--artifact_base_folder", "-a", type=str, default="./artifacts/breast_hist_images",
                        help="Dataset directory.")
    parser.add_argument("--data_split_dir", "-f", type=str, default="/home/arne/datasets/SICAPv2/partition/Validation/Val1/",
                        help="path to the directory of data lists")
    parser.add_argument("--save_model", "-s", type=str, default="models/simple_cnn.h5",
                        help="Path to save model.")
    parser.add_argument("--load_model", "-l", type=str, default="None",
                        help="Path to .h5 model to load for testing or retraining. Set to 'None' for new model.")
    parser.add_argument("--output_dir", "-o", type=str, default="output/",
                        help="Path to .h5 model to load for testing or retraining. Set to 'None' for new model.")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--image_target_size", "-i", type=Tuple, default=(96, 96),
                        help="Image resolution after rescaling")
    args = parser.parse_args()
    main(args)