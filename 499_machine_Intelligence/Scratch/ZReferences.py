#This is a single line commment

'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 1
'''


#print("Hello World!")
#print("Goodbye")

#Example of input number with arithmatic
#fTemp = float(input("What's the temperature (in Fahrenheit)? "))
#cTemp = (fTemp - 32) * (5 / 9)
#print("It's now " + str(cTemp) + " C")

age = int(input("How old are you"))

if age < 32:
    print ("You young girl")
else:
        print("You old mommas...")
        print ("WOOOOW")


print ("Baiiii")


vowels = 'aeiou'
name = input("What's your name: ")
while name:
    my_vowels = [v for v in name if v.lower() in vowels]
    print(len(my_vowels))
    name = input("What's your name: ")

oldstring = 'I like Guru99'
newstring = oldstring.replace('like', 'love')
print(newstring)

def Convert(string):
    li = list(string.split(" "))
    return li

# Driver code
str1 = "Geeks for Geeks"
print(Convert(str1))

cals_consumed = 1500
cals_in_item = 100
total_cals = cals_consumed + cals_in_item

print (f"totalColaroies: {total_cals}")

if total_cals>=1600:
    print("Dont eat it!")

elif total_cals > 300:
    print("Eat a larger portion.")

else:
    print("sure you can eat!")


words = ['i','hate','cows']
new_words = map(lambda x: 'you' if x == 'i' else x, words)
print (new_words)  #prints ['you', 'hate', 'cows']


'''
convention…. Python generally considers the under bar to be easier to read and therefore more “pythonic”

This is advanced word replacement, All it does is return a tuple for each iteration… the tuple contains the index and the value.

it’s a PEP8 standard.
and constants like BAD_WORDS are all caps. Variables are lower case.
'''

BAD_WORDS = ("dren", "frak", "frel", "glob", "grud", "narf", "zark")
​
while True:
    text_input = input('What shall I censor? ')
    if not text_input:
        break
    list_of_words = text_input.split()
    for idx, val in enumerate(list_of_words):
        if val.lower() in BAD_WORDS:
            list_of_words[idx] = 'BEEP'
    print(' '.join(list_of_words))
​
print("Goodbye!")


word = ['dren', 'frak', 'frel', 'glob', 'grud,', 'narf', 'zark']
def censor(text, word):
    empty_text = text.split(" ")
    for e, i in enumerate(empty_text):
        for x in word:
            if i == x:
                empty_text[e] = "BLEEP"
            else:
                pass
    result = " "
    print (result.join(empty_text))

while True:
    text = input('Enter a sentence (or press Enter to quit): ')
    if not text:
        break
    else:
        censor(text, word)
print("Goodbye")


while True:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    shift = int(input("Enter the number to shift by (0-25): "))
    if True:
        if shift < 0 or shift > 25:
            print("You must pick a number between 0-25")
            continue
        if shift > 0 or shift < 25:
            message = input("Enter a message: ")
            encryptedList = [ ]
            encryptedAlphabet = alphabet[shift:] + alphabet[:shift]
            for letter in message:
                index = alphabet.index(letter)
                encryptedList.append(encryptedAlphabet[index])
            encryptedMSG = "".join(encryptedList)
            print("Encrypting message....")
            print("Encrypted message:", encryptedMSG)
            encryptedList2 = [ ]
            encryptedAlphabet = alphabet[shift:] + alphabet[:shift]
            for letter in encryptedMSG:
                index = encryptedAlphabet.index(letter)
                encryptedList2.append(alphabet[index])
            encryptedMSG2 = "".join(encryptedList2)
            print("Decrypting message....")
            print("Decrypted message:", encryptedMSG2)


'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Assignment 4 – Chuck-a-luck
'''

#Part 1: Setup
import random                                                                       #We are importing random so that we can use the random functionality
import time                                                                         #For Dramatic effect, I want to import time so that I can delay a print statement later

def roll():                                                                         #Returns a list of 3 integers holding the results of the dice rolls
    randomNumbers = []                                                              #Holds the 3 random numbers generated
    for x in range(3):                                                              #This is for the range of numbers we want to hold, we want three so we want a range of
        x = random.randint(1, 6)                                                    #We want integers between 1-6
        randomNumbers.append(x)                                                     #We then append the number into the random number list
    #print(randomNumbers)                                                            #Just used to test that this function was working correctly

def computeBetResult():                                                             #0)Ingests 3 users inputs
    a, b, c = [int(x) for x in input
    ("PLease enter 3 integers (separated each integers by a guess): ").split()]     #1) The integer list containing the 3 numbers representing the dice rolls
    amountToBet = int(input("How much would you like to bet? "))                    #2) A single integer containing the amount the user bet
    whatDidyaGuess = int(input("What did ya guess? "))                              #3) A single integer holding the number the user “guessed”
    amountOfMoneyUserWon = 0                                                        #This is the amount of money the user won (placeholder)?
    #print(a, b, c)                                                                  #Just used to test that this function was working correctly
    #print(a)                                                                        #Just used to test that this function was working correctly
    #print(b)                                                                        #Just used to test that this function was working correctly
    #print(c)                                                                        #Just used to test that this function was working correctly
    #print(amountToBet)                                                              #Just used to test that this function was working correctly
    #print(whatDidyaGuess)                                                           #Just used to test that this function was working correctly
    #print(amountOfMoneyUserWon)                                                     #Just used to test that this function was working correctly

def main ():

    #Part 2: User interface

    print("Hey there, Welcome to my game....")
    print("Chuck-A-Luck (table game) is a lively game where three "
          "dice tumble in a spinning cage and you place wagers on "
          "how many dice will come up with your chosen number when "
          "the cage stops spinning. This isn't that game, but same "
          "concept. Lets do it.")

    usersMoney = int(100)
    howMuchUserBets = int(input("You have " + str(usersMoney) + " dollars. How much would you like to bet?: "))
    while howMuchUserBets:
        if howMuchUserBets == 0:
            break
        else:
            if (0 > howMuchUserBets) or (howMuchUserBets <= usersMoney):
                numberUserBetsOn = int(input("Please select a number to bet on (between 1 and 6): "))
                if (numberUserBetsOn > 6) or (numberUserBetsOn == 0):
                    numberUserBetsOn = int(input("Not a valid number, Please select a number to bet on "
                                                 "(between 1 and 6, 0 to quit): "))
                    continue
                else:
                    print("Okie dokie, you are betting " + str(howMuchUserBets) + " dollars and you are betting on #" + str(numberUserBetsOn) + ".")
                    break
            else:
                howMuchUserBets = int(input("This bet is invalid...you have " + str(usersMoney) + " dollars. How much would you like to bet?: "))
    roll()
    computeBetResult()

main()

'''
Resources:
-Generating random integers
    https://stackoverflow.com/questions/33224944/generate-a-list-of-6-random-numbers-between-1-and-6-in-python/33224964
    https://stackoverflow.com/questions/49356294/simple-lottery-generate-new-six-unique-random-numbers-between-1-and-49
    https://www.geeksforgeeks.org/random-numbers-in-python/
    https://pythonprogramminglanguage.com/randon-numbers/
-List input integer
    https://stackoverflow.com/questions/4663306/get-a-list-of-numbers-as-input-from-the-user
'''
