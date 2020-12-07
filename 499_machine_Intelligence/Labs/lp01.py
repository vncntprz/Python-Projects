'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 1
'''

#This is just an intro for fun and added
print ("Welcome to MadLibs 1.0.")

#This is where the user puts in thier name
storeName = input("What is your name? ")
storedName = storeName.strip()

#This is where we store the information on the favorite 4 legged animal
storefourLeggedAnimal = input("What's your favorite 4 legged animal (in plural form)? ")
storedFourLeggedAnimal = storefourLeggedAnimal.strip()

#This is where we store information on thier favorite number
storeNumber = int(input("Enter your favorite number? ") )

#This is where we calculate the funny number that goes into the madlib
funnyNumber = ((storeNumber * 4)+2)

madLibintro = "Here's your MadLib:"

print ('\n' + madLibintro)
print(storedName.title() + " led" + " " + str(storeNumber) + " " + "\"" + storedFourLeggedAnimal.title() + "\"" + ".")
print ("They all left" + " " + str(funnyNumber)+ " " + "footprints each step of the way!")

