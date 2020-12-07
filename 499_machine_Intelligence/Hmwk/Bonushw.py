'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
BonusHomework
'''


def genRoster(roster):  # process file function
    #fileopen = open(roster, "r")  # It opens a file and reads in all the numbers in the file into a number list
    #roster_read = fileopen.read().splitlines()  # It opens a file and reads in all the numbers in the file into a number list
    while roster is True:
        if roster == "roster.txt":
            break
        if roster != "roster.txt":
            print("There's a problem with the file", roster)
            roster = input("Try again, where are the grades?")
            continue

def main():
    roster_dictionary = {}
    roster = input("Where are the grades?: ").lower()
    genRoster(roster)

main()
