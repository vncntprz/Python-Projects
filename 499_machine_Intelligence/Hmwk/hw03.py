'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Assignment 3 – Calendar
'''

#Part 1: Months dictionary

monthsDictionary = {"january": 31, "february": 28, "march": 31,             #This dictionary contains the month and the number of days within that month. We will use these for checks in the code later.
                    "april": 30, "may": 31, "june": 30, "july" : 31,
                    "august": 31, "september": 30, "october": 31,
                    "november": 30, "december": 31,}

values = list(monthsDictionary.values())                                    #This is to check the values (the month names) in the months dictionary
keys = list(monthsDictionary.keys())                                        #this is to check the keys (the number of days) in the months dictionary

#Part 2: Calendar dictionary
calendar = {}                                                               #We call an empty dictionary so that we can create empty lists for ecah day of the month
for key in dict.keys(monthsDictionary):                                     #This is where we initialize the key in the dictionary to popul ate in the new calendar dictionary
    calendar[key] = 0                                                       #We then give the dictionary values for each month a value of 0
for key, value in monthsDictionary.items():                                 #Here is where we populate empty lists in leiu of the 0 value above
    calendar[key] = [''] * value                                            #We multiply the empty list by the number of days in the months dictionary.

#Part 3: User input
while True:                                                                 #While this statement is true, meaning while we submit an answer
    inputDictionary = {}                                                    #we initialize an empty dictionary to hoold the iniputs outside of the loops
    userInput = input("Enter month and day (EX: January 31): ").lower()     #We ask for the user input in a specific format and we make that input into lowercase
    if len(userInput.split()) > 2:                                          #if the length of the input is greater than 2, then it prints out an error message
        print("Not a valid entry")                                          #we print this
        continue                                                            #We give the user the opportunity to redo and we take them back to the top of loop
    if not userInput:                                                       #If the user puts nothing, then we break the loop and go directly to the ending print statements
        break
    month, day = userInput.split()                                          #We place the split input into two variables, month and day
    month = month                                                           #We then place month again in its own variable line
    day = int(day)                                                          #And we make the day into an integer
    inputDictionary[month] = day                                            #We then put the month and day into the input dictionary up above, though I dont think I utilize this at all
    if month not in monthsDictionary:                                       #if the month is not in the month dictionary above, then we inform the user that this is not a valid month
        print("Not a valid month...")
        continue                                                            #The user has the opportunity to try again and we take them to the top of the loop
    if month in monthsDictionary:                                           #If the month is valid, we want to check that the number of days is correct
        if monthsDictionary[month] < day:                                   #If the integer is greater than the number of days, then we inform the user
            print("This is not a valid date, that month only has",
                  monthsDictionary[month], "days...")
            continue                                                        #We then direct the user to the top of the loop to try again
    holiday = input("What happens on this day? ")                           #if the user is successful, meaning the day, month, and input is valid, we then ask what happened on this day
    for month in inputDictionary:                                           #We then reference two dictinoaries, the input and the months dictionary,
        if (monthsDictionary[month] >= day) and (month in monthsDictionary):#We double check to ensure the entry is valid by confirming the month and the day are valid
            calendar[month][day - 1] = holiday                              #We then append the holdiay to the empty list in the calendar up above

#Part 4: Display results
print("")                                                                   #Just a space holder to break up the text
for month, day in calendar.items():                                         #We then reference the month and day, the month being the month, and the day being the empty list
    for index, value in enumerate(calendar[month]):                         #We index the empty list number and give it a value with enumerate
        if value != "":                                                     #We want to ignore the empty lists
            print(month.title(), index+1, ":", value)                       #We then print the month, the list number along with the value of the string in that list
print("")                                                                   #Just a space holder to break up the text
print("Goodbye!")                                                           #Goodbye.
