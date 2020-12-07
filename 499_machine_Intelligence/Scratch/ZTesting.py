#
#      Benjamin R. Carpel
#      ACAD 499, Spring 2020
#      carpel@usc.edu
#      Assignment 3
#

# PART 1 HERE (Month dictionary)

monthDict ={"January":31, "February":28, "March":31, "April":30, "May":31, "June":30, "July":31, "August":31, "September":30, "October":31, "November":30, "December":31}
print(monthDict)

month_name = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

# PART 2 HERE:  Empty to be filled calendar dictionary
for month in monthDict:
    dateList =[]
    for day in range(1,monthDict[month]+1,1 ):
        dateList.append(str(day))
#    print(dateList)  #this is a test to make sure this wa working

# PART 3 HERE: User input

while True:
    user_input = input('Hi there! Please enter a date (format example = "July 4"): ')

    if len(user_input.split()) != 2:
        if len(user_input.split()) < 2:
            print()
            break
        else:
            print("That format doesn't work")
            continue

    month, day = user_input.split()
    month = month.capitalize()

    if day.isnumeric():
        day = int(day)

    if (month in month_name) and (day in range(1, len(monthDict[month])+1)):
        event_input = input("What happens on {user_input}? ")
        monthDict[month][day-1] = event_input

    elif (month not in month_name) or (day not in range(1, len(monthDict[month])+1)):
        if month not in month_name:
            print("Nah that doesn't work '{month}'")
        else:
            print(
                "That month only has {len(monthDict[month])} days")
    else:
        break
    print()


for name in month_name:

    for index, string in enumerate(monthDict[name]):
        if string != "":
            print("{name} {index+1} : {string}")







'''# PART 1 HERE (Month dictionary)
monthDict ={"January":31, "February":28, "March":31, "April":30, "May":31, "June":30, "July":31, "August":31, "September":30, "October":31, "November":30, "December":31}
print(monthDict)
month_name = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
# PART 2 HERE:  Empty to be filled calendar dictionary
for month in monthDict:
    dateList =[]
    for day in range(1,monthDict[month]+1,1 ):
        dateList.append(str(day))
#    print(dateList)  #this is a test to make sure this wa working
# PART 3 HERE: User input
while True:
    user_input = input('Hi there! Please enter a date (format example = "July 4"): ')
    if len(user_input.split()) != 2:
        if len(user_input.split()) < 2:
            print()
            break
        else:
            print("That format doesn't work")
            continue
    month, day = user_input.split()
    month = month.capitalize()
    if day.isnumeric():
        day = int(day)
    if (month in month_name) and (day in range(1, len(monthDict[month])+1)):
        event_input = input("What happens on {user_input}? ")
        monthDict[month][day-1] = event_input
    elif (month not in month_name) or (day not in range(1, len(monthDict[month])+1)):
        if month not in month_name:
            print("Nah that doesn't work '{month}'")
        else:
            print(
                "That month only has {len(monthDict[month])} days")
    else:
        break
    print()
for name in month_name:
    for index, string in enumerate(monthDict[name]):
        if string != "":
            print("{name} {index+1} : {string}")


#PARTS 1 and 2 HERE
cal_dictionary = {
    "January": [""], "February": [""],
    "March": [""], "April": [""], "May": [""],
    "June": [""], "July": [""], "August": [""],
    "September": [""], "October": [""], "November": [""], "December": [""]
}
month_list = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
# Fill this up
for days_count, months in zip(days_in_month, month_list):
    cal_dictionary[months] *= days_count
#PART 3 HERE
while True:
    user_input = input('Hi there! Please enter a date like this format: "July 1" (without quotes): ')
    if len(user_input.split()) != 2:
        if len(user_input.split()) < 2:
            print()
            break
        else:
            print("Sorry, that input format doesn't work")
            continue
    month, day = user_input.split()
    month = month.capitalize()
    # convert to number and make sure the day is numeric
    if day.isnumeric():
        day = int(day)
    # len(cal_dictionary[month]) returns the number of days in that month
    if (month in month_list) and (day in range(1, len(cal_dictionary[month])+1)):
        event_input = input("What happens on " + user_input + "?")
        cal_dictionary[month][day-1] = event_input
    elif (month not in month_list) or (day not in range(1, len(cal_dictionary[month])+1)):
        if month not in month_list:
            print("This month '{month}' isn't there, can you try again?")
        else:
            print(
                "Close, but that month only has {len(cal_dictionary[month])} days")
    else:
        pass
    print()
#PART 4 HERE
for name in month_list:
# my friend gave me a hint and said 'enumerate' brings pairs for a count + value He also mentioned "f strings" for
# formatting the result in a concise easy manner, I guess?
    for index, string in enumerate(cal_dictionary[name]):
        if string != "":
            print(f"{name} {index + 1} : {string}")
'''
