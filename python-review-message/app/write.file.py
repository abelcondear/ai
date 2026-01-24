import sys
import datetime

# append message value received by parameter
# through command line
with open("review.message.log", "a") as file:
    now = datetime.datetime.now()

    [today, hhmm] = str(now).split(" ")

    [yyyy, mm, dd] = today.split("-")
    [hh, mmm, ss] = hhmm.split(":")

    sss = ( "0" + str(int(float(ss))) )[-2:]
    cur_datetime = f"{dd}-{mm}-{yyyy} {hh}:{mmm}:{sss}"

    file.write\
    (
        f"This message {sys.argv[1:][0]} "
        f"has priority level. [ {cur_datetime} ]"
    )
    file.write("\n")
