import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def welcome():
    print("======= Salary Predicion System =======")
    print("Press Enter key to proceed !!")
    input()
def checkcsv():
    csv_files=[]                      #list ke andhar har ek csv files hoga
    curr_dir=os.getcwd()              #getcwd()= crrent working directory batata hai
    content_list=os.listdir(curr_dir)
    for x in content_list:
        if x.split('.')[-1] =='csv':
            csv_files.append(x)
    if len(csv_files)==0:
        return "No CSV Files in directory"
    else:
         return csv_files
def display_and_select_csv(csv_files):
    i=0
    for filename in csv_files:
            print(i,'.',filename)
            i+=1
    return csv_files[int(input("Select the csv file to create an ML Model"))]

def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_predict):
    plt.scatter(X_train,Y_train,color="red",label="Training data")
    plt.plot(X_train,regressionObject.predict(X_train),color="blue",label="Best Fit")
    plt.scatter(X_test,Y_test,color="green",label="Test data")
    plt.scatter(X_test,Y_predict,color="yellow")
    #plt.scatter(X_test,Y_predict,color="orange",label="Ypredict")
    plt.title("Expectations of Salary")
    plt.xlabel("Years of experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()


def main():
    welcome()
    try:
        csv_files=checkcsv()
        if csv_files == "No CSV Files in directory":
            raise FileNotFoundError("No csv files found in directory..")
        csv_file = display_and_select_csv(csv_files)
        print(csv_file,'is been selected')
        print("Reading Dataset.............")
        print("Creating Dataset.............")
        dataset= pd.read_csv(csv_file)
        print("Dataset has been created successfully!!!!!")

        #Splitting Training and Testing data

        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        s=float(input("Enter the test data size (between 0 and 1) "))
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=s)
        print("Model Creation in progress......")

        #Making of Regression Line using training data

        regressionObject = LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Model creation has been done.")
        print("Press enter key to predict test data in train model")
        v=input().split(',')
        print(v)

        #Using testing data to predict Ypredict values

        Y_predict = regressionObject.predict(X_test)

        #checking ml model

        print("Experience", " ...", "Actual Salary", " ...", "Predicted Salary")
        i=0
        while i<len(X_test):
            print(X_test[i], "          ",Y_test[i],"         ",Y_predict[i] )
            i+=1

        #To see in Graph

        print("Press enter key to see above result in graphical format")
        input()
        graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_predict)

        #Model performance(r2)
        r2= r2_score(Y_test,Y_predict)
        print("Our model is %2.2f%% accurate"  %(r2*100))
        print("Now you can predict the salary of an employee using our model...")
        print("\nEnter experience in years of the candidates, separated by comma")
        exp = [float(e) for e in input().split(',')]      #list comprehension
        ex=[]
        for x in exp:
            ex.append([x])
        experience = np.array(ex)
        salaries = regressionObject.predict(experience)
        plt.scatter(experience,salaries,color='red')
        plt.xlabel("Experience")
        plt.ylabel("Salary")
        plt.show()
        d = pd.DataFrame({'Experience': exp, 'Salaries': salaries})
        print(d)
    except FileNotFoundError:
            print("No csv file found in the directory")
            print("Press Enter key to exit")
            input()
            exit()
if __name__ == "__main__":
    main()
    input()               #main fxtn khatam hone ke baad koi bhi key press karo to end

