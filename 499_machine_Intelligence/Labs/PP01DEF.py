'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 1
'''

def main ():
    answerYes = "y"
    answerNo = "n"
    questionMath = input("Do you want to do some math, y or n:")
    #questionMathAnswer = questionMath.strip(())
    while questionMath.lower() == answerYes:
        print("Please put in two numbers")
        firstNumber = float(input("First number:"))
        x = firstNumber
        secondNumber = float(input("Second number:"))
        y = secondNumber
        answerOp = input("Would you like to subtract, add, multiply, or divide:")
        if answerOp == "subtract":
            answerSub = subtract(x,y)
            print("Here is your subtraction answer:", answerSub)
        elif answerOp == "add":
            answerAdd = add(x,y)
            print("Here is your addition answer:", answerAdd)
        elif answerOp == "multiply":
            answerMulti = multiply (x,y)
            print("Here is your multiplication answer:", answerMulti)
        elif answerOp == "divide":
            if secondNumber == 0:
                print("Cannot divide by zero")
            else:
                answerDivide = divide (x,y)
                print("Here is your division answer:", answerDivide)
        else: print("You need to select an operation")
        questionMath = input("Want to do math again, y or n:")
    print("sianara!")
def divide (x, y):
    return (x/y)
def multiply (x, y):
    return (x * y)
def add (x, y):
    return (x+y)
def subtract (x, y):
    return(x-y)
main()
