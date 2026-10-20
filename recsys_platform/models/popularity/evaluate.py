from recsys_platform.cli import parse_evaluation_args, run_evaluation

from .model import PopularityRecommender


if __name__ == "__main__":
    args = parse_evaluation_args("PopularityRecommender")
    run_evaluation(args, recommender_cls=PopularityRecommender)
