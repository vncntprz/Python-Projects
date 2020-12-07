'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Assignment 4 – Chuck-a-luck
'''


# Part 1: Setup
import random  # We are importing random so that we can use the random functionality
import time  # For Dramatic effect, I want to import time so that I can delay a print statement later

def roll():   # Returns a list of 3 integers holding the results of the dice rolls
    randomNumbers = []  # Holds the 3 random numbers generated
    for x in range(3):  # This is for the range of numbers we want to hold, we want three so we want a range of
        x = random.randint(1, 6)  # We want integers between 1-6
        randomNumbers.append(x)  # We then append the number into the random number list
    return randomNumbers

def computeBetResults (rolls, betAmt, guessNumber): #The function should return the payout based on how many times the user’s number showed upon the dice.
    count = 0
    for roll in rolls: #Go through the inputted dice rolls and determine how many times the bet upon number occurs. It should be a number between 0 and 3.
        if roll == guessNumber:
            count += 1
    print("You matched", count, "dice!")
    if count == 0:
        return -betAmt #If the number occurred 0 times,the payout is 0.You should return negative their bet amount as they lost that money.
    elif count ==1:
        return betAmt #If the number occurred 1 time, payout is 1 times the amount of money the userbet (plus their original bet amount).
    elif count ==2:
        return betAmt *3 #If the number occurred 2 times, payout is 3 times the amount of money the user bet (plus their original bet amount).
    else:
        return betAmt * 10 #If the number occurred 3 times, payout is 10 times the amount of money the user bet plus their original bet amount).

def main (): #Introduces the user to the game, announces how much money they have, and asks for an amount to bet (which you should store in another integer variable).
    # Part 2: User interface
    print("Hey there, Welcome to my game....")
    print("Chuck-A-Luck (table game) is a lively game where three "
        "dice tumble in a spinning cage and you place wagers on "
        "how many dice will come up with your chosen number when "
        "the cage stops spinning.")
    time.sleep(3)
    print("This isn't that game, but same concept. Lets do it.")
    money = 100 #variable to hold the user’s “money”
    print("You have", money, "Dollars to use for betting...") #Tells the user their new money total.
    playAgain = "y"
    while playAgain.lower() == "y":
        while True:
            bet = int(input("How much money do you want to bet?: ")) #Here the user inputs how much they want to bet
            if bet <= 0 or bet > money: #We double check if the bet is greater than or equal to 0 or more than money had
                print("Invalid bet. Your bet must be more than 0 and less than the money you have...")
                continue
            break
        while True:
            userguess = int(input("Please select a number to bet on (between 1 and 6): ")) #This is where the user inputs thier money
            if userguess < 1 or userguess > 6: #This is to check if the number the user bet on is valid
                print("Invalid bet. Your bet must be more than 0 and less than the money you have...")
                continue
            break
        rolls = roll() #This is where we store the users rolls from the roll function
        print("You rolled: ")
        print(str(rolls[0])+","+ str(rolls[1])+","+str(rolls[2])) #This makes the list of the dice rolls look like numbers instead of a printed listed
        print("You bet", bet, "Dollars on", userguess) #This summarizes how much the user bet and on what number
        winnings = (computeBetResults(rolls, bet, userguess))
        money += winnings #This adds the new winnings to the total dollars the user has
        print("Your have", money, "dollars, you can do better...") #This then prints out the new total of money the user has
        if money == 0: #Here the game checks if the user has no money, if no money, the program breaks.
            print("You are out of money, they house won. Try again next time...")
            break
        playAgain = input("Play again? You know you want to. (Y or N): ") #Ask the user if they’d like to play again. If the user enters “N” or “n”, stop the game. The gameshould also stop if the user runs out of money.
    print("Good thing this isn't vegas")

main() #This is where we run


'''
Resources:
-Generating random integers
    https://stackoverflow.com/questions/33224944/generate-a-list-of-6-random-numbers-between-1-and-6-in-python/33224964
    https://stackoverflow.com/questions/49356294/simple-lottery-generate-new-six-unique-random-numbers-between-1-and-49
    https://www.geeksforgeeks.org/random-numbers-in-python/
    https://pythonprogramminglanguage.com/randon-numbers/
'''
