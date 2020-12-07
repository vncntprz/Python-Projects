'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 2
'''



#This is where we get the input from the user.
nameEntered = input("What is your name (enter / return to quit)? ")

#This is where the program calculates the length of the name inputed.
lengthNameEntered = len(nameEntered.strip())

#This is the value of 0 when length of name entered.
answerNo = len(nameEntered.strip()) == 0
answerYes = len(nameEntered.strip()) > 0

#This is where we define the vowels to reference later.
vowels = set("aeiouAEIOU")

while answerYes:

    #These are just the values of where I need the counter to be and the placeholder for the vowels to sit when the calculation is done.
    numVowels = 0
    zero = 0

    #This resets the value of numvowels to 0 just in case in the previous run the value isnt what I need it to be.
  #  if numVowels > 0:
      #  numVowels = (numVowels * zero)

    #This is to check length of input and ensure that the length of the nameEntered is greater than 0.
    #if answerYes is True:

        #Dont fuck with this vincent (3 lines down prints the vowels).
    for letter in nameEntered.strip():
         if letter in vowels:
            numVowels += 1

        #This prints how many vowels are in the input.
    print("That name has " + str(numVowels) + " vowels!")

    #This restarts the loop
    nameEntered = input("What is your name (enter / return to quit)? ")

    #This is to exit the program.
    if not nameEntered:
        break

#HOLY SHIT THAT WAS A BRAIN TEASER.
