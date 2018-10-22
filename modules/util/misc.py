"""Miscellaneous utils"""


def yn_prompt(question):
    """
    Ask yes no question.

    Args:
        question:

    Returns: True if yes, False if no.

    """
    yes = {'yes','y', 'ye'}
    no = {'no', 'n'}
    if not question.endswith(" "):
        question += " "
    print(question, end="")
    while True:
        choice = input().lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond with 'yes' or 'no': ", end="")