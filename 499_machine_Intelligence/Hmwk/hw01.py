'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Homework 1
'''

#This is where we ask the user for thier name so we can use in the function.
userName = input("Hello, whose BMI shall I calculate? ")

#This is the question that prompts to enter two different variables for the users Height.
print ("Okay first I need, " + userName.strip() +"'s height. I'll take it in feet and inches. ")

#This is where we get the users height specifically.
userHeightFeet = int(input("Feet First... "))
userHeightInches = int(input("Now inches... "))

#This is where we convert the users hieght from feet into inches and add the remaining inches to the equation
userOverallHeightInInches = ((userHeightFeet * 12) + userHeightInches)
userOverallHegihtInMeters = float(userOverallHeightInInches / 39.3701)

#this is where we prompt continuing conversation to get weight.
print("Thanks. Now I need " + userName.title() +"'s weight in pounds.")

#This is where we get the users weight specficially.
userWeight = float(input("Please enter " + userName.title() + "'s weight "))

#This is where we calculate user weight in Kilo.
userWeightInKilo = float(userWeight / 2.20462)

#This is where we calculate user BMI.
userBMI = float(userWeightInKilo / ((userOverallHegihtInMeters)**2))
hundredsBMI= userBMI * 100
integerBMI = int(hundredsBMI)
finalBMI = integerBMI /100.0

#This is where the BMI is printed
print(userName + "'s BMI is " + str(finalBMI) +".")

#This is just to check the statements are printing correctly.
#print("")
#print("Inputs and calculations:")
#print(userName)
#print(userHeightFeet)
#print(userHeightInches)
#print(userWeight)
#print(userOverallHeightInInches * 12)
#print(userOverallHegihtInMeters)
#print(userWeightInKilo)
#print(finalBMI)

#def truncateBMI ():
 #   hundredsBMI= userBMI * 100
  #  integerBMI = int(hundredsBMI)
   # finalBMI = integerBMI /100.0
    #print(finalBMI)
