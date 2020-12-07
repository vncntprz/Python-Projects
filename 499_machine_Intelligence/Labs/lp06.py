'''
Vincent Perez, MS-IDBT Cohort 6 - 5847842892
ACAD 499 Machine Intelligence / AI for Design Applications, Spring 2020
vincenjp@usc.edu
Lab practical 6 -
'''

def processFile(): #process file function
    numberfile = input("Number file name: ").lower()
    fileopen = open(numberfile, "r") #It opens a file and reads in all the numbers in the file into a number list
    numbersread = fileopen.read().splitlines() #It opens a file and reads in all the numbers in the file into a number list
    numberslist = [] #Empty list
    for number in numbersread: #It returns a list of integers as output
        numberslist.append(int(number)) #Appends the numbers in the document as integers
    #print(numberslist)
    fileopen.close() #We close the file
    return numberslist #Returns the number list

def getAverage(numbers): #It accepts a list of integers as input
    average = sum(numbers)/len(numbers)
    #print(sum(numbers))
    #print(len(numbers))
    #print("Average = ", average)
    return average #Returns average from the list

def maximum(numbers): #It accepts a list of integers as input
    maximum = max(numbers) #It returns the maximum value of the list (biggest value in the list)
    #print("Max = ", maximum)
    return maximum #Returns maximum of list

def minimum(numbers): #It accepts a list of integers as input
    minimum = min(numbers) #It returns the minimum value of the list (smallest value in the list)
    #print("Min = ", minimum)
    return minimum #Returns minimum of list

def main():
    numbers = processFile() #Process the document and returns the numbers in the list
    print("Maximum:", maximum(numbers)) #Prints Maximum def function
    print("Minimum:", minimum(numbers)) #Prints Minimum def function
    print("Average:", getAverage(numbers)) #Prints average number function

main()
