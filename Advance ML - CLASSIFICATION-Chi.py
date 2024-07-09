def selectkbest(indep_x,dep_y,n):
    test=SelectKBest(score_func=chi2,k=n)
    fit1=test.fit(indep_x,dep_y)
    selectk_features=fit1.transform(indep_x)
    return selectk_features
    
def split_scalar(indep_x,dep_y):
    x_train,x_test,y_train,y_test=train_test_split(indep_x,dep_y,test_size=0.25,random_state=0)
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    return x_train,x_test,y_train,y_test

def cm_prediction(classifier,x_test):
    y_pred=classifier.predict(x_test)
#Making confusion matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import classification_report
    accuracy=accuracy_score(y_test,y_pred)
    report=classification_report(y_test,y_pred)
    return classifier,accuracy,report,x_test,y_test,cm

def logistic(x_train,y_train,x_test):
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def svm_linear(x_train,y_train,x_test):
    from sklearn.svm import SVC
    classifier=SVC(kernel="linear",random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def svm_NL(x_train,y_train,x_test):
    from sklearn.svm import SVC
    classifier=SVC(kernel="rbf",random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def Navie(x_train,y_train,x_test):
    from sklearn.naive_bayes import GaussianNB
    classifier=GaussianNB()
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def knn(x_train,y_train,x_test):
    from sklearn.neighbors import KNeighborsClassifier
    classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def DecisionTree(x_train,y_train,x_test):
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier(criterion="entropy",random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def RandomForest(x_train,y_train,x_test):
    from sklearn.ensemble import RandomForestClassifier
    classifier=RandomForestClassifier(n_estimators=10, criterion="entropy",random_state=0)
    classifier.fit(x_train,y_train)
    classifier,accuracy,report,x_test,y_test,cm=cm_prediction(classifier,x_test)
    return classifier,accuracy,report,x_test,y_test,cm

def selectK_classification(acclog,accsvm1,accsvmn1,accknn,accnav,accdes,accrf):
    dataframe=pd.DataFrame(index=["chiSquare"],columns=["Logistic","SVM1","SVMn1",
                                                        "KNN","Navie","Decision","Random"])
    for number,idex in enumerate(dataframe.index):
        dataframe["Logistic"][idex]=acclog[number]
        dataframe["SVM1"][idex]=accsvm1[number]
        dataframe["SVMn1"][idex]=accsvmn1[number]
        dataframe["KNN"][idex]=accknn[number]
        dataframe["Navie"][idex]=accnav[number]
        dataframe["Decision"][idex]=accdes[number]
        dataframe["Random"][idex]=accrf[number]
    return dataframe    