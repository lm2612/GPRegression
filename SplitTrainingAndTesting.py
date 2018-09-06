def split_set(X,y,names,TestNames):
    """ Manually select test data to split for training and testing """
    assert (all([TestName in names for TestName in TestNames])),'All scenarios listed in TestNames must be in names. Check for typos. TestNames= {0}. names={1}'.format(TestNames,names)
    idx_test = [names.index(TestName) for TestName in TestNames]
    idx_train = [idx for idx in range(len(names)) if idx!=idx_test]
    X_train,X_test = X[idx_train],X[idx_test]
    y_train,y_test = y[idx_train],y[idx_test]
    names_train = [names[idx] for idx in idx_train]
    names_test = [names[idx] for idx in idx_test]
    return(X_train,X_test,y_train,y_test,names_train,names_test)


def split_set_random(X,y,names,test_number,seed):
    """ Splits set into a random test set and training set based on seed,
    with number of test sets being test_number and training sets being N-test_number """
    (N,p) = X.shape
    assert (test_number<N),"Select test_number to be less than total number of samples
    in set. test_number={0}, number of samples={1}".format(test_number,N)
    idxs = np.arange(N)
    (X_train,X_test,y_train,y_test,idx_train,idx_test) = train_test_split(X,y,idxs,test_size=np.float(test_number)/np.float(N),random_state=seed)
    names_train = [names[idx] for idx in idx_train]
    names_test = [names[idx] for idx in idx_test]
    return(X_train,X_test,y_train,y_test,names_train,names_test)
