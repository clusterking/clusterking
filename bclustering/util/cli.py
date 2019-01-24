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


if __name__ == "__main__":
    # for testing
    ans = yn_prompt("This is a test.")
    print(ans)
