Practical Machine Learning Project
=============================================

-------------------------------------------

##Background

In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

The training dataset has 19622 observations of 160 variables, while test dataset has 20 observations of 160 variables. 

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set.  


```r
train = read.csv("pml-training.csv",header=TRUE)
test = read.csv("pml-testing.csv",head=TRUE)
```

##Preprocessing

#### 1.First we check the status of missing values of the dataset.


```r
train2 = train[,8:160]
mm = data.frame(index=rep(0,152), percent = rep(0,152))
for(i in 1:152){
  mm$index[i] = i
  mm$percent[i] = ifelse((length(which(is.na(train2[,i])|train2[,i]==""))/nrow(train2)) >.1, 0, 1)
}
sub_mm = mm[which(mm$percent > 0 ),]
nrow(sub_mm)
```

```
## [1] 52
```

```r
train2 = train2[,c(sub_mm[,1],153)]
test_new = test[,8:160][,c(sub_mm[,1])]
```

The code above first find out the number of missing values of each predictors. I first remove the first 7 columns unrelated for the analysis and then the ones with more than 10% percent of missing values as fixing these variables would cause great bias to the model. After removing the variables with too many(more than 10%) missing values, we now have 52 predictors without missing values left for the following analysis. 

#### 2. Check the balance status of observations


```r
table(train2[,60])
```

```
## Error: undefined columns selected
```

The tables shows that the number of cases "A" is more than other 4, but not that much. Therefore, there is no need to do balance adjustment.


##Modeling

- In this part, we begin to build the classifier for further prediction. There are 2 models I used to build the classifier: CART and Random Forest. 

- To aviod overfitting, I use cross validation in every model and set the fold as 4. Therefore , it is a 4-folds cross validation. And Finally we will select the one with lowest prediction error rates, which is also "out of sample error", to do the prediction.

- Before modeling, I randomly set 80% cases in the training set to modeling and 20% left to test the model. By this, I can get the prediction error of the model.

#### 1. CART


```r
library(caret)
library(rattle)

set.seed(666)
index = createDataPartition(train2$classe, p=0.2, list=FALSE)
training = train2[index,]
testing = train2[-index,]

cart.fit = train(classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = training, method="rpart")

cart = cart.fit$finalModel
fancyRpartPlot(cart)
```

![plot of chunk cart](figure/cart.png) 

```r
cart.fit$resample
```

```
##   Accuracy  Kappa Resample
## 1   0.5061 0.3517    Fold1
## 2   0.5148 0.3666    Fold4
## 3   0.4975 0.3431    Fold3
## 4   0.5046 0.3539    Fold2
```

```r
p = predict(cart.fit, newdata = testing)
paste("prediction error = ",1-sum(diag(table(p,testing$classe)))/nrow(testing))
```

```
## [1] "prediction error =  0.479834342147181"
```

As we can see, the prediction error rate is a bit high in the model CART with Cross Validation. Then we use Random Forest.


#### 2.Random Forest


```r
rf.fit = train(classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = training, method="rf")
rf.fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 2.47%
## Confusion matrix:
##      A   B   C   D   E class.error
## A 1112   3   1   0   0    0.003584
## B   25 724  10   1   0    0.047368
## C    0  17 661   6   1    0.035036
## D    1   2  15 625   1    0.029503
## E    0   4   4   6 708    0.019391
```

```r
p = predict(rf.fit, newdata = testing)
paste("prediction error = ",1-sum(diag(table(p,testing$classe)))/nrow(testing))
```

```
## [1] "prediction error =  0.0331952851226506"
```

As we can see, Random Forest does perfect in this case with very low prediction error. 

Thus I will use Random ForesT to do the prediction.


## Predction


```r
p = predict(rf.fit, newdata = test_new)
p
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

