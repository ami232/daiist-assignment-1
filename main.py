"""
Entry point required by the assignment: `uv run python main.py train` runs
the training pipeline, `uv run python main.py app` launches the Gradio
dashboard. Looks for <stage>.py first, then <stage>.ipynb, at the repo root.

NOTE: check whether your forked repo's template already provides a
main.py that does this -- if so, keep whichever one you actually use,
don't run two dispatchers. This is here in case you need one.
"""

import runpy
import sys
from pathlib import Path

VALID_STAGES = ["train", "app"]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_STAGES:
        print(f"Usage: python main.py {{{'|'.join(VALID_STAGES)}}}")
        sys.exit(1)

    stage = sys.argv[1]
    py_path = Path(f"{stage}.py")
    ipynb_path = Path(f"{stage}.ipynb")

    if py_path.exists():
        runpy.run_path(str(py_path), run_name="__main__")
    elif ipynb_path.exists():
        # Running a notebook end-to-end without manual steps.
        import nbformat
        from nbclient import NotebookClient

        nb = nbformat.read(str(ipynb_path), as_version=4)
        client = NotebookClient(nb)
        client.execute()
    else:
        print(f"Neither {py_path} nor {ipynb_path} found.")
        sys.exit(1)


if __name__ == "__main__":
    main()