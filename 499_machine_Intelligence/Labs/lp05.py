'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 5
'''

'''
Use a dictionary to store each letter in the alphabet and a corresponding frequency. 
The alphabet should use uppercase letters as the keys and integer counters for the values. 
You will use one entry for both uppercase and lowercase letters (for example “A” will be the key for both “a” and “A”.
• HINT: You can use a loop to create the initial the key-value pairs. 
Using ord and chr in a loop you can generate all the keys and initialize their values to 0.
• Repeatedly ask the user for text. When the user enters no text, the input loop should end.
• Display the letter frequency from the user’s text as a histogram with “*” to indicate each letter
occurrence. The histogram must display the letters in alphabetical order, but omit letters with no occurrences.
• HINT: Because the keys are not necessarily stored in ascending in the dictionary, you should first
generate a list of all the keys. Then use the list’s sort function to get the keys in the correct order.
'''

#import time                                                            #I just really like delaying the print statements
                                                                        #Creation of Alphabet list, Alphabet Dictionary with value of 0
alphalist = []                                                          #This is our empty list where we generate all the letters in the alphabet
alpha = 'A'                                                             #We want the characters to be uppercase, so we start with capital "A"
for character in range(0, 26, 1):                                       #We want each character in the alpahabet range
    alphalist.append(alpha)                                             #We append each new letter to the list
    alpha = chr(ord(alpha) + 1)                                         #and in succession, we add a caps letter after the previous before it.
alphaDict = dict.fromkeys(alphalist, 0)                                 #This was beyond me, but somehow we are able to use the fromkeys input from the list of letters and give them all a value of 0.
                                                                        #This is where we initialize the phrase so we can begin counting.
frequency = {}                                                          #We want to hold the number of letters held in the phrases we enter somewhere, so it will be stored here.
while True:                                                             #While the input is true it will run the while statement
    gimmeAPhrase = (input("Gimme a phrase boi...: ")).upper()           #This is where we begin the loop to take in the input
    gimmeAPhraseToList = list(gimmeAPhrase)                             #We convert that input into a list
    if not gimmeAPhrase:                                                #If the input is not entered (we do return instead)
        break                                                           #Then we break
    else:                                                               #If there is an entered value
        for letter in gimmeAPhraseToList:                               #For each letter in the input list
            if letter in frequency:                                     #We check i its in the frequency dictionar
                frequency[letter] += 1                                  #If it is in the frequency dictionary, we add 1 to that dictionary key
            else:
                frequency[letter] = 1                                   #If not in the frequency dictionary, we add the key and then set the value to 1
        for letter in frequency:                                        #If the letter is in the frequncey dictionary,
            if letter in alphaDict.keys():                              #We cross reference the alpha dictionary
                alphaDict[letter] = frequency[letter]                   #We then change the value of the dictionary key to the frequency of the input
print("Calculating the frequency of the words you've entered...")       #Makes the user think the computer is "calculating" thier response
for key, value in alphaDict.items():                                    #For each key, we seperate each value as a list item
    #time.sleep(3)                                                      #This is just for fun, I want to delay the print frequency for effect,
    print(key, ' : ', "*" * value)                                      #We then print and multiply the asterisk by the value in the values dictionary
print("Seeeee Ya! Maybe try counting your own letters next time?")

'''
#Test Prints
print(alphalist)
print(alphaDict)
print(gimmeAPhraseToList)
print(frequency)
'''

'''
Resources:
-Initiallizing alphabets
    https://www.geeksforgeeks.org/python-ways-to-initialize-list-with-alphabets/
-Convert a list to a dictionary
    https://thispointer.com/python-how-to-convert-a-list-to-dictionary/
-Upper Lower comments
    https://www.geeksforgeeks.org/isupper-islower-lower-upper-python-applications/
-How to String a list of characters
    https://stackoverflow.com/questions/4978787/how-to-split-a-string-into-array-of-characters
    https://thehelloworldprogram.com/python/python-string-methods/
-Python List count
    https://www.programiz.com/python-programming/methods/list/count
-Counter
    https://www.tutorialspoint.com/counting-the-frequencies-in-a-list-using-dictionary-in-python
    https://pymotw.com/2/collections/counter.html
-Dictionaries
    https://www.w3schools.com/python/python_dictionaries.asp
'''
