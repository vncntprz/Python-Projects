'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 3
'''

#   0       1       2       3       4       5       6
# "dren", "frak", "frel", "glob", "grud", "narf", "zark"
#   -7      -6      -5      -4      -3      -2      -1

badWords = ["dren", "frak", "frel", "glob", "grud", "narf", "zark"]             #I've defined the words we want to censor in this list
delimiter = " "                                                                 #and I have created a delimiter for when I want to concatenate the list back together via " ".
while True:                                                                     #While this value is true,
    censorInput = input("What shall I censor: ")                                #The value that is true is the input for censorInput
    censorInputLower = censorInput.lower()                                      #once we get an input, we then make it lower so I only have to reference the list once and only lowercase
    censorInputToList = list(censorInputLower.split())                          #Here is where we make the inputt into a list via the split function, but note we are making the lower list into a function
    if not censorInput:                                                         #If there is no input, this will break the loop.
        break
    else:                                                                       #If there is an input, then it proceeds to the for statement
        for i in censorInputToList:
            if i in badWords:                                                   #If there is a badword is contained in censorInputList via variable I,
                indexPos = censorInputToList.index(i)                           #we note the index position of that bad word
                censorInputToList[indexPos] = "BLEEP"                           #we then find and replace that word with the word "BLEEP".
        newOutputString = delimiter.join(censorInputToList)                     #This is for the new output string with the new words replaced
        print(newOutputString)                                                  #Finally we print the new phrase.
print("Goodbye!")                                                               #If the while statement is broken, then it prints goodbye!
