from recsys_platform.cli.train import get_training_parser, run_training

from .als_trainer import ALSTrainer, ALSTrainerConfig
from .bpr_trainer import BPRTrainer, BPRTrainerConfig
from .lmf_trainer import LMFTrainer, LMFTrainerConfig


ALGORITHMS = {
    "ALS": {"config_cls": ALSTrainerConfig, "trainer_cls": ALSTrainer},
    "BPR": {"config_cls": BPRTrainerConfig, "trainer_cls": BPRTrainer},
    "LMF": {"config_cls": LMFTrainerConfig, "trainer_cls": LMFTrainer},
}


if __name__ == "__main__":
    parser = get_training_parser("Matrix factorization type recommender")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ALS", "BPR", "LMF"],
        default="ALS",
        help="Algorithm used to create the model (default: %(default)s)",
    )

    args = parser.parse_args()

    kwargs = ALGORITHMS[args.algorithm]
    run_training(args, **kwargs)
