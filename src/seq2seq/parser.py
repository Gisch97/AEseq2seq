import argparse
from ._version import __version__
def parser():   

    parser = argparse.ArgumentParser(
        prog="seq2seq",
        description="Autoencoder sequence-to-sequence: an end-to-end method for RNA sequence prediction based on deep learning",
        epilog="webserver link | None",
    )
    parser.add_argument("-c", "--config", type=str, help="Config file (optional, overrides any other options)")
    parser.add_argument("-d", type=str, default="cpu", help="Device ('cpu' or 'cuda')")
    parser.add_argument("-batch", type=int, default=4, help="Batch size for handling sequences")
    #parser.add_argument("--use-restrictions", action="store_true", help="Use external restriction matrix (default: False)")
    parser.add_argument("-j", type=int, default=2, help="Number of parallel workers")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (default: False)") 
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length to process (default: 512") 
    parser.add_argument('--version', '-v', action='version', version='%(prog)s ' + __version__)

    subparsers = parser.add_subparsers(
        title="Actions", dest="command", description="Running commands", required=False
    )

    parser_train = subparsers.add_parser("train", help="Train a new model")
    parser_train.add_argument(
        "train_file",
        type=str,
        
        help="Training dataset (csv file with 'id', 'sequence')",
    )

    parser_train.add_argument(
        "--valid-file",
        type=str,
        help="Validation dataset to stop training. If not provided, validation split is randomly generated from training data. Columns are the same as training",
    )
    parser_train.add_argument(
        "-o",
        type=str,
        dest="out_path",
        help="Output path (if not provided, it is generated with the current date)",
    )

    parser_train.add_argument(
        "-n", "--max-epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true", help="Cache of data for training (default: cache is used)",
    )

    # test parser
    parser_test = subparsers.add_parser("test", help="Test a model")
    parser_test.add_argument(
        "test_file",
        type=str,
        help="Testing dataset (csv file with 'id', 'sequence')",
    )
    parser_test.add_argument(
        "-w", type=str, dest="model_weights", help="Trained model weights"
    )
    parser_test.add_argument(
        "-o",
        type=str, dest="out_path", 
        help="Output test metrics (default: only printed on the console)",
    )

    # pred parser
    parser_pred = subparsers.add_parser(
        "pred", help="Predict structures for a list of sequences"
    )
    parser_pred.add_argument(
        "--name", type=str, default="console_input", dest="sequence_name", help="Sequence name (default: console_input)"
    )

    parser_pred.add_argument(
        "pred_file",
        type=str,
        help="Dataset to predict. It can be a csv file with 'id' and 'sequence' columns or a fasta file",
    )
    parser_pred.add_argument(
        "-o",
        type=str, dest="out_path", 
        help="Output path, it can be a .csv file or a directory to hold CT files or interaction score files  (default: pred.csv)",
    )

    parser_pred.add_argument(
        "-w", type=str, dest="model_weights", help="Trained model weights"
    )

    return parser.parse_args()
    