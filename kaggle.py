import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from keras.layers import Dropout
from keras import optimizers

train = pd.read_csv("data/test.csv", sep=',')
label = pd.Series(train["time"].values,name = "time")
def data_preprocessing(df):
    jobs_max = np.max(df["n_jobs"])
    df.loc[df["n_jobs"] != -1, "n_jobs"] = np.log(df.loc[df["n_jobs"] != -1, "n_jobs"])
    df.loc[df["n_jobs"] == -1, "n_jobs"] = np.log1p(jobs_max)

    df.loc[df["penalty"] == "none", "l1_ratio"] = 0
    df.loc[df["penalty"] == "l2", "l1_ratio"] = 0

    df.loc[df["penalty"] == "none", "penalty"] = np.log(2)
    df.loc[df["penalty"] == "l2", "penalty"] = np.log(2)
    df.loc[df["penalty"] == "l1", "penalty"] = np.log(4)
    df.loc[df["penalty"] == "elasticnet", "penalty"] = np.log(4)

    df["info_ratio"] = df["n_informative"]/df["n_features"]
    df["noise"] = df["n_samples"] * df["flip_y"]

    col = df.columns
    scaled_df = StandardScaler().fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=col)
    return df

train = data_preprocessing(train)
predictors = ["penalty","max_iter","n_jobs","n_samples","n_features", "info_ratio","noise","n_classes","n_informative","alpha","l1_ratio"]
X = train[predictors]
y = train["time"]

X,y = shuffle(X,y,random_state = 1)
row = train.shape[0]
X_train,y_train = X[:row - 100],y[:row - 100]
X_test,y_test = X[row:],y[row:]

def hidden_layer(model, n_neuron, leaky_alpha):
    #model.add(Dropout(0.2))
    model.add(Dense(n_neuron, kernel_initializer='he_normal',activation = "linear"))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_alpha))

# define the model
def create_model(input_size):
    leaky_alpha = .005
    # create model
    model = Sequential()
    model.add(Dense(input_size, input_dim=input_size, kernel_initializer='he_normal' ))

    hidden_layer(model, 8, leaky_alpha)
    hidden_layer(model, 8, leaky_alpha)
    #hidden_layer(model, 16, leaky_alpha)
    hidden_layer(model, 8, leaky_alpha)
    hidden_layer(model, 4, leaky_alpha)
    hidden_layer(model, 2, leaky_alpha)

    model.add(Dense(1, kernel_initializer='he_normal'))
    # Compile model
    op = optimizers.Adam(lr=0.007, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0000001, amsgrad=True)
    model.compile(loss='mean_squared_error', optimizer=op)
    return model

model = create_model(len(predictors))
#training
print("Training-----------")
cost = 1000
tol = eval(input("Please input tolerance:"))
step = 0
while cost > tol:
    cost = model.train_on_batch(X_train, y_train)
    if step % 100 == 0:
        print('Step:',step,'.train cost:', cost)
    step += 1

#test
print("\nTesting----------")
cost = model.evaluate(X_test,y_test,batch_size = 100)
print('test cost:',cost)

#predict
f_name = input("Please input file name:")
test = pd.read_csv("data/test.csv", sep=',')
test = data_preprocessing(test)
result = model.predict(test[predictors]).ravel()
test["time"] = pd.Series(result)
sub = test[["id","time"]]
sub.to_csv("data/submission_" + f_name +".csv",index=False)
