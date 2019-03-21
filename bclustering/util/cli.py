#!/usr/bin/env python3

"""Utils for the command line interface (CLI)."""


def yn_prompt(question: str, yes=None, no=None) -> bool:
    """Ask yes-no question.

    Args:
        question: Description of the prompt
        yes: List of strings interpreted as yes
        no: List of strings interpreted as no

    Returns:
        True if yes, False if no.
    """
    if not yes:
        yes = ['yes', 'ye', 'y']
    if not no:
        no = ['no', 'n']

    prompt = question
    if not prompt.endswith(" "):
        prompt += " "
    prompt += "[{} or {}] ".format('/'.join(yes), "/".join(no))

    print(prompt, end="")

    while True:
        choice = input().lower().strip()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond with '{}' or '{}': ".format(
                '/'.join(yes), "/".join(no)), end="")


def handle_overwrite(paths, behavior, log):
    """ Do we want to overwrite a file that exists?

    Args:
        paths: List of pathlib.Paths
        behavior: How to proceed if output file already exists:
            'ask', 'overwrite', 'raise'
        log: logging.Logger instance

    Returns:
        True if overwrite will occurr, False otherwise.
    """
    behavior = behavior.lower()
    if any([p.exists() for p in paths]):
        if behavior == "ask":
            prompt = "Some of the output files would be overwritten. " \
                     "Are you ok with that?"
            if not yn_prompt(prompt):
                log.warning("Returning without doing anything.")
                return
        elif behavior == "overwrite":
            pass
        elif behavior == "raise":
            msg = "Some of the output files would be overwritten."
            log.critical(msg)
            raise FileExistsError(msg)
        else:
            msg = "Unknown option for 'overwrite' argument."
            log.critical(msg)
            raise ValueError(msg)
        return True
    return False


if __name__ == "__main__":
    # for testing
    ans = yn_prompt("This is a test.")
    print(ans)
