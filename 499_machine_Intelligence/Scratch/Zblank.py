#Part 3: User input


while True:
    monthAndDateInput = input("Enter month and day (EX: January 31): ")
    monthAndDateInputLower = monthAndDateInput.lower()
    monthAndDateInputToList = monthAndDateInputLower.split()
    if not monthAndDateInput:
        break
    else:
        month, date = monthAndDateInputToList[0],  monthAndDateInputToList[1]
        if month in calendar.keys():
            print("This Month is one of the keys in the calendar dictionary...")
        if month not in calendar:
            print("This Month is not one of the keys in the calendar dictionary...")
            month = (input("Please enter a month: "))

print("Goodbye!")



'''  
print(monthAndDateInput)
print(calendar.values())
print(calendar.keys())
'''
#calendar[key][find the inidex (what day in month) monthAndDateInputToList[1]]
if len < 2, input again
    reference the months dictionary
    if value >< enter proper value

    check month spelling


#Create a list that has 12 items, each item consists of a empty list and corresponds to the number of days in the month, assign first value in that calendar with the value inputed with for in loop
for value in dict.values(monthsDictionary):
    calendar[value]


while True:
    monthAndDateInput = input("Enter month and day (EX: January 31): ")
    monthAndDateInputLower = monthAndDateInput.lower()
    monthAndDateInputToList = monthAndDateInputLower.split()
    if not monthAndDateInput:
        break
    else:
        month, date = monthAndDateInputToList[0],  monthAndDateInputToList[1]
        if month in calendar.keys():
            print("This Month is one of the keys in the calendar dictionary...")
        if month not in calendar:
            print("This Month is not one of the keys in the calendar dictionary...")
            month = (input("Please enter a month: "))

print("Goodbye!")



'''  
print(monthAndDateInput)
print(calendar.values())
print(calendar.keys())
'''
#calendar[key][find the inidex (what day in month) monthAndDateInputToList[1]]
if len < 2, input again
    reference the months dictionary
    if value >< enter proper value

    check month spelling
calendar =  {
    "January": [],"February":[], "March":[], "April":[], "May":[], "June":[], "July":[], "August":[], "September":[], "October":[], "November":[], "December":[]}


#Create a list that has 12 items, each item consists of a empty list and corresponds to the number of days in the month, assign first value in that calendar with the value inputed with for in loop
#for value in dict.values(monthsDictionary):
    #calendar[value]



#calendar[key] = list * value
'''x = 3 # amount of lists you want to create
for i in range(1, x+1):
    command = "" # this line is here to clear out the previous command
    command = "list" + str(i) + " = []"
    exec(command)'''


'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Assignment 3 – Calendar
'''

#Part 1: Months dictionary

monthsDictionary = {"january": 31, "february": 28, "jarch": 31, "april": 30, "may": 31, "june": 30, "July" : 31, "august": 31, "september": 30, "october": 31, "november": 30, "december": 31,}
values = list(monthsDictionary.values())
keys = list(monthsDictionary.keys())

#Part 2: Calendar dictionary
calendar = {}
for key in dict.keys(monthsDictionary):
    calendar[key] = 0
for key, value in monthsDictionary.items():
    calendar[key] = [''] * value

#Part 3: User input
while True:
    monthAndDateInput = input("Enter month and day (EX: January 31): ")
    monthAndDateInputLower = monthAndDateInput.lower()
    monthAndDateInputToList = monthAndDateInputLower.split()
    monthAndDateInputLength = len(monthAndDateInputToList)
    if not monthAndDateInput:
        break
    else:
        while monthAndDateInput:
            if monthAndDateInputLength <2:
                print ("Looks like youre missing a space or didnt enter a day, try again.")
                continue
            elif monthAndDateInputToList[0] not in dict.keys(monthsDictionary):
                print ("Month can't be found, try again.")
                monthAndDateInputToList[0] = input("Enter month (EX: January): ")
                continue
            elif (monthAndDateInputToList[1] in dict.values(monthsDictionary)) and (monthAndDateInputToList[1] not in range(1, len(monthsDictionary[values]))):
                print ("Date can't be found, try again.")
                monthAndDateInputToList[0] = input("Enter Date (EX: 31): ")
                continue
    whatHappened = input("What happens on January, 1?")
print("Goodbye!")

#Part 4: Display results
