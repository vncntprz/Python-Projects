import random

#This is the list of words to be scrambled/jumbled. The Theme is Jungle
wordList = ["tiger", "lions", "monkey", "zebra", "snake"]

#print(letters)

while True:

    #This uses random.choice to randomly select word from list and creates new variable
    chosenWord = (random.choice(wordList))

    #Lists all the letters in chosenWord, in order
    letters = list(chosenWord)
    scrambledletters = ''

    #This prints the letter in index position 3
    #print(letters[3])
    #This replaces the letter in index positions with an "i"
    letterSwap = letters[0]
    letters[0] = letters[4]
    letters[4] = letterSwap
    letterSwap = letters[1]
    letters[1] = letters[2]
    letters[2] = letterSwap
    letterSwap = letters[2]
    letters[2] = letters[0]
    letters[0] = letterSwap
    print("The Jumbled Word Is: " + scrambledletters.join(letters))
    answer = input("Please Enter Your Guess: ") #this ask the user to guess
    guess = answer.lower()
    count = 0
    usertries = ("It Took You " + str(count) + " Tries")
    if not guess:
        break
    else:
        if guess == chosenWord:
            print(usertries)
        while guess != chosenWord:
            count = count + 1
            answer = input("Try Again. Please Enter Your Guess: ") #this ask the user to guess
            guess = answer.lower()
            if guess == chosenWord:
                print("It Took You " + str(count) + " Tries")
            continue

print("Goodbye!")
