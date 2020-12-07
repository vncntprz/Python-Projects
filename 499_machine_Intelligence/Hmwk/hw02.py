'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Assignment 2 – Word Jumbled and Encryption
'''


#Part 1 – Word Jumble Game
import random
wordBank = ["cat", "dog", "bird", "lizard", "octopus", "elephant", "otter",
            "python", "corgi", "mongoose", "snake", "caterpillar", "kangaroo"]                  #This is our wordbank that the program can choose from.

while True:                                                                                     #While this statement is true, the statement being the input from the user
    randomWord = random.choice(wordBank)                                                        #The program will choose a random word
    position = random.randrange(len(randomWord))                                                #Here the program will determine a random number from the length of the random word
    jumbledword = randomWord[position:] + randomWord[:position]                                 #We then slice the word to give the appearance of jumble
    count = 1                                                                                   #Our counter is set to 1, so that if a user gets the answer right the first time, its correct already
    randomWordOutputOption = input("Unscramble this word, " + "'" + jumbledword + "'" + ": ")   #This is the prompt for the user to enter thier guess of the random word
    if not randomWordOutputOption:                                                              #If the user presses enter or does not guess, we break or stop the program
        break
    else:                                                                                       #If the user then answers
        if randomWordOutputOption == randomWord:                                                #If correct, the program tells the user how many times it took the user to guess correctly
            print("You got that in " + str(count) + " go!")
        while randomWordOutputOption != randomWord:                                             #if the user guesses wrong, a whileloop is started until the guess is correct
            randomWordOutputOption = input("Try again: ")
            count +=1                                                                           #We then tally the number of guesses the user took to get the right answer
            if randomWordOutputOption == randomWord:                                            #If correct, the program tells the user how many times it took the user to guess correctly
                print("You got that in " + str(count) + " go(s)!")
            continue
print("See ya later alligator!")                                                                #This is the response when a user does not guess at all. 

#this is just space between programs.
print("")
print("")
print("")
print("")
print("")
print("")

#Part 2 – Encrypt / Decrypt
#Note: There are times that the program errors out, please restart program if this happens. Myabe Pycharm?

while True:                                                                 #While the message varaible is true, this program will run
    alphabet = "abcdefghijklmnopqrstuvwxyz"                                 #The Alphabet string we refernce our message to
    encryptedList = [ ]                                                     #This is just the placeholder for the encrypted list to be placed in
    message = input("Enter a message: ")                                    #We grab the user input here
    if not message:                                                         #If message not true, then we break
        break
    else:
        shift = int(input("Enter the number to shift by (0-25): "))         #We ask the user by how many positions should we shift the message
        if shift < 0 or shift > 25:                                         #we want to ensure the user only picks a number between 0-25, so we check here
            print("You must pick a number between 0-25")                    #If the user picks a number less than 0 or more than 25, they recieve this message
        if shift > 0 or shift < 25:                                         #If the user picks an appropriate number, then the program continues
            encryptedAlphabet = alphabet[shift:] + alphabet[:shift]         #This is where we take the shift number input and slice and past the shift to the end of the encrypted alphabet
            for letter in message:                                          #For every letter in the message,
                index = alphabet.index(letter)                              #We index each position so that we can swap out the letter with the encrypted alphabet
                encryptedList.append(encryptedAlphabet[index])              #We then apply the encrytped alphabet to the message
            encryptedMSG = "".join(encryptedList)                           #And we join the message together with the new alphabet
            print("Encrypting message....")                                 #This is just placeholder tex to appear as if its "working"
            print("Encrypted message:", encryptedMSG)                       #Here we display the encrypted message with the shifted alphabet
            encryptedList2 = [ ]                                            #This is just the placeholder for the encrypted list 2 to be placed in
            encryptedAlphabet = alphabet[shift:] + alphabet[:shift]         #We then take the same prinicipal by applying the actual alphabet to the encrypted message
            for letter in encryptedMSG:                                     #where we decrypt each letter with the alphabet key string
                index = encryptedAlphabet.index(letter)                     #Here we index each encrypted letter and then decrypt with the alphabet
                encryptedList2.append(alphabet[index])                      #We append the original alphabet to the original encrypted message
            encryptedMSG2 = "".join(encryptedList2)                         #Here the encrypted message get rejoined together from string form
            print("Decrypting message....")                                 #This is just placeholder tex to appear as if its "working"
            print("Decrypted message:", encryptedMSG2)                      #Here we display the encrypted message with the decrypted message with the regular alphabet
print("Goodbye!")                                                           #If the user presses enter, then the program prints this to end.
