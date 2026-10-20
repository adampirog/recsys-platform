from recsys_platform.cli import parse_training_args, run_training

from .trainer import PopularityTrainer, PopularityTrainerConfig


if __name__ == "__main__":
    args = parse_training_args("PopularityRecommender")
    run_training(args, config_cls=PopularityTrainerConfig, trainer_cls=PopularityTrainer)
