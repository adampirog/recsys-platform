from recsys_platform.cli.evaluate import get_evaluation_parser, run_evaluation

from .model import PopularityRecommender


if __name__ == "__main__":
    parser = get_evaluation_parser("PopularityRecommender")
    run_evaluation(parser.parse_args(), recommender_cls=PopularityRecommender)
