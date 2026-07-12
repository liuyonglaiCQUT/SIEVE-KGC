"""Convenience wrapper: run filtered evaluation and save top-10 predictions."""
import sys

from evaluate import main


if __name__ == "__main__":
    if "--save_predictions" not in sys.argv:
        sys.argv.extend(["--save_predictions", "true"])
    main()
