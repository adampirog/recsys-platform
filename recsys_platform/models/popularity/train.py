from recsys_platform.cli.train import get_training_parser, run_training

from .trainer import PopularityTrainer, PopularityTrainerConfig


if __name__ == "__main__":
    parser = get_training_parser("PopularityRecommender")
    run_training(
        parser.parse_args(),
        config_cls=PopularityTrainerConfig,
        trainer_cls=PopularityTrainer,
    )
