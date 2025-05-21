def main():
    from utils.args import build_parser
    from utils import fine_tune

    parser = build_parser()
    args = parser.parse_args()

    if args.operation == "fine-tune":
        fine_tune()


if __name__ == '__main__':
    main()