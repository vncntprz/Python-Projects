
wantToPlay = input ("Want to play a game? Enter y or n.").lower()

if (wantToPlay == "y"):
    #need to tell the user the rules of the game
    myNumber = random.randrange(10) + 1

    #let the user guess the number
    userNumber = int(input("What's your guess?"))

    numberIsGuessed = False

    white not (numberIsGuessed):
        if myNumber == userNumber:
            #they guessed it
            numberIsGuessed = True

            while not(numberIsGuessed):
                if myNumber == userNumber
                    numberIsGuessed = True

                else:
                        print ("You didnt guess my number. Try Again.")
                        userNumber = int(input("Whats your guess?"))


            print ("You guessed my number")
