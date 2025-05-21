import argparse

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal example: parses various arguments."
    )

    # Category 1: General Arguments
    ge_group = parser.add_argument_group('General Arguments')
    ge_group.add_argument(
        '-op', '--operation',
        choices=['fine-tune', 'synthesize', 'predict'],
        required=True,
        help='Mode of operation: train, eval, or predict (default: train).'
    )
    ge_group.add_argument(
        '-out', '--output',
        type=str,
        default='ouptut',
        help='Output directory for the results.'
    )
    ge_group.add_argument(
        '--config',
        type=str,
        help='Path to the configuration file.'
    )

    # Category 2: Data Arguments
    data_group = parser.add_argument_group('Data Arguments')
    data_group.add_argument(
        '--input',
        help='Path to the input data file.'
    )
    data_group.add_argument(
        '--batch-size',
        help='Batch size for training.',
        type=int
    )

    # Category 3: Training Arguments
    train_group = parser.add_argument_group('Training Arguments')
    train_group.add_argument(
        '--deepspeed-config',
        type=str,
        help='Path to the DeepSpeed configuration file.'
    )

    return parser

def get_args_by_group(parser: argparse.ArgumentParser, args: argparse.Namespace, group_title: str) -> dict:
    """
    Returns a dict of argument names and their values for the specified argument group.
    """
    for group in parser._action_groups:
        if group.title == group_title:
            return {action.dest: getattr(args, action.dest, None) for action in group._group_actions}
    return {}