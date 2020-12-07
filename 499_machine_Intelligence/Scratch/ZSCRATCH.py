#Part 1: Months dictionary

#This dictionary contains the month and the number of days within that month. We will use these for checks in the code later.
monthsDictionary = {"january": 31, "february": 28, "march": 31, "april": 30, "may": 31, "june": 30,
                    "july" : 31, "august": 31, "september": 30, "october": 31, "november": 30, "december": 31,}
values = list(monthsDictionary.values())                                    #This is to check the values (the month names) in the months dictionary
keys = list(monthsDictionary.keys())                                        #this is to check the keys (the number of days) in the months dictionary

#Part 2: Calendar dictionary
calendar = {}                                                               #We call an empty dictionary so that we can create empty lists for ecah day of the month
for key in dict.keys(monthsDictionary):                                     #This is where we initialize the key in the dictionary to popul ate in the new calendar dictionary
    calendar[key] = 0                                                       #We then give the dictionary values for each month a value of 0
for key, value in monthsDictionary.items():                                 #Here is where we populate empty lists in leiu of the 0 value above
    calendar[key] = [''] * value                                            #We multiply the empty list by the number of days in the months dictionary.

#Part 3: User input
while True:
    inputDictionary = {}
    userInput = input("Enter month and day (EX: January 31): ").lower()
    if len(userInput.split()) > 2:
        print("Not a valid entry")
        continue
    if not userInput:
        break
    month, day = userInput.split()
    month = month
    day = int(day)
    inputDictionary[month] = day
    if month not in monthsDictionary:
        print("Not a valid month...")
        continue
    if month in monthsDictionary:
        if monthsDictionary[month] < day:
            print("This is not a valid date, that month only has", monthsDictionary[month], "days...")
            continue
    holiday = input("What happens on this day? ")
    for month in inputDictionary:
        if (monthsDictionary[month] >= day) and (month in monthsDictionary):
            calendar[month][day - 1] = holiday

#Part 4: Display results
print("")
for month, day in calendar.items():
    for index, value in enumerate(calendar[month]):
        if value != "":
            print(month.title(), index+1, ":", value)
print("")
print("Goodbye!")
