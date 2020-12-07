'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 4
'''

import random                                                               #We are importing random so that we can use the.random functionality
import time                                                                 #For Dramatic effect, I want to import time so that I can delay a print statement later
valuesDictionary = {}                                                       #This is where we store the values for the dictionary
for value in range(2, 52, 1):                                               #Since we didnt type out the values for the dictionary, we used a range instead between 1-51
    valuesDictionary[value] = 0                                             #This is the value for the dictionary items
n = int(input("Pick a number between 1 and 10,000: "))                      #We ask for the input here for an integer between 1-10,000
selectedRandomNumbers = []                                                  #This is where we store the selected random numbers for use later
for x in range(n):                                                          #The value of x is just a placeholder as we do not know the range or the value yet
    selectedRandomNumbers.append(random.randrange(0, 10001, 1))             #This is where we append the random selected numbers to the empty list
for number in selectedRandomNumbers:                                        #This is where we are able to reference the number of selected random numbers and begin calculating the frequencies
    for divisor in range(2,52,1):                                           #If the divisor is between 1-51 with a step of 1,
        if number % divisor == 0:                                           #If the number is divisible by the range
            valuesDictionary[divisor] +=1                                   #We add 1 to the values dictionary value
print("You random numbers are: " + str(selectedRandomNumbers))              #We show the user selected numbers here
time.sleep(5)                                                               #For dramatic effect, we pause for 5 seconds before moving onto the next print statement
print("For " + str(n) + " random numbers, are the factor frequencies...")   #This is just a print statement to break up the lines
for key, value in valuesDictionary.items():                                 #For each key, we seperate each value as a list item
    print(key, ' : ', "*" * value)                                          #We then print and multiply the asterisk by the value in the values dictionary
