import sys

with open("review.message.log", "a") as file:
    file.write(f"This message {sys.argv[1:][0]} has priority level.")
    file.write("\n")

