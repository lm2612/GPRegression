def split_set(X,y,names,TestName):
    idx_test = [names.index(TestName)]
    idx_train = [idx for idx in range(len(names)) if idx!=idx_test]
    X_train,X_test = X[idx_train],X[idx_test]
    y_train,y_test = y[idx_train],y[idx_test]
    names_train,names_test = names[idx_train],names[idx_test]
    return(X_train,X_test,y_train,y_test,names_train,names_test)
