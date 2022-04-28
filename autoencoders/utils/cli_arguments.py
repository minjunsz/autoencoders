import argparse


def get_arguments():
    """configure argparser

    Returns:
        Dict: given cli arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configFile', help="config yaml file name", default='config.yaml')
    args = parser.parse_args()
    return args
