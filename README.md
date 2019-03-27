# Intro to Machine Learning

Brought to you by Galvanize. Learn more about the way we teach at [galvanize.com](http://galvanize.com).


Learn more Python & Data Science with Galvanize Data Science Premium Prep! Currently free if you sign up now! [http://bit.ly/2u0cahU](http://bit.ly/2u0cahU)

Get to this repo by typing in URL: **ml.sage.codes**

### FAQ: 

- WIFI: `Galvanize Guest Seattle` | Password is posted on the wall
- Bathrooms: Behind you down the hall to the left
- Kitchen outside back classroom door with Coffee & Tea!
- Snacks + water in back of room


## Setting up your computer
* A web browser to see what we're working on as others see it (Recommend Google Chrome: [chrome.google.com] (http://chrome.google.com))
* We will be using Google Colab for this workshop so make a Google accoung if you don't already have one. 
* Open this github Repo to follow along


# What this workshop is

A super friendly introduction to Machine Learning No previous experience expected, but knowing some python will help!

You can't learn EVERYTHING in ~2 hours, especially when it comes to Machine Learning! But you can learn enough to get excited and comfortable to keep working and learning on your own! 

- This course is for absolute beginners
- Ask Questions!
- Answer Questions!
- Help others when you can
- Its ok to get stuck, just ask for help!
- Feel free to move ahead
- Be patient and nice

We're not going to focus on the math behind the models. We're going to focus more on when and how to use a model. If you would like to go into the math and more about each model I encourage you to do so!

## About me:

Hello I'm [Sage Elliott](http://sageelliott.com/). I'm a Technology Evangelist here at Galvanize! For the past decade I've worked as a software and hardware engineer with Startups and Agencies in Seattle, WA and Melbourne, FL. I love making things with technology! 

I Originally got into Machine Learning by solving a manufacturing problem at my last job with computer vision, and I think its one of the coolest fields!

**Note:** I'm not a Galvanize Instructor, they're way better at teaching than I am!

- Website: [sageelliott.com](http://sageelliott.com/)
- Twitter: [@sagecodes](https://twitter.com/@sagecodes)
- LinkedIn: [sageelliott](https://www.linkedin.com/in/sageelliott/) 
- Email: [sage.elliott@galvanize.com](mailto:sage.elliott@galvanize.com)

Reach out to me if interested in:

- breaking into the tech industry 
- learning resources
- meetup recommendations 
- learning more about Galvanize
- giving me suggestions for events!
- being friends


## About you!

Give a quick Intro!

- Whats your name?
- Whats your background?
- Why are you interested in Machine Learning?



# What is Machine Learning:


To put it very simply Machine Learning can usually be thought of using a statistical model built based on a dataset to solve a problem. 

Instead of explicitly programming an algorithm to do a specific task, we let it "learn" from data to find patterns and inference.

We'll see example of this soon!


###  Who uses Machine Learning?

More and more companies using making descions with data are using machine learning. Here are just a few examples that you've probably experiences as a customer.

#### Amazon

- Product Recommendations
- Amazon GO Computer Vision
- Alexa 
- Delivery Robots


#### Netflix

- Show & Movie Recommendations

#### Google

- Gmail Spam Filtering
- Google Assistance
- Youtube Content filtering & Recommendations
- Self Driving Cars

#### Apple

- Siri
- App Store Recommendations

#### Facebook

- Face Tagging Detection

#### Tesla

- Self Driving Cars

These companies use Machine Learning in many other ways!



### Machine Learning Applications

We talked about a some examples above from big companies we probably all know of. But here are several more types of applications that machine learning has become popular with.

#### Healthcare

- Cancer Detection
- X-Ray diagnostic 

#### Smart Home Devices

- Smart door Beel
- Smart Lights
- Security

#### Image generation

- [NVIDIA’s Hyperrealistic Face Generator](https://medium.com/syncedreview/gan-2-0-nvidias-hyperrealistic-face-generator-e3439d33ebaf)
- video game Character or level generation
- [art generation](https://www.artnome.com/news/2018/3/29/ai-art-just-got-awesome)


#### Agriculture

- Crop monitoring & planning

#### Supply Chain 

- Sourcing and Shipping Automation

#### Manufacturing 

- Quality Assurance
- Design


#### Fraud Detection

- Credit cards
- Product listings

You can see how all of these applications revolve around finding patterns in data!



# Types of Machine Learning:


## Supervised Learning

Supervised Learning uses a dataset that is labeled. In this context imagine having a list of features and 


Here we have features(sepal length (cm), etc) and a label(Flower Species)

| sepal length (cm)  | sepal width (cm)  | petal length (cm)  | petal width (cm)  |  Species | 
|---|---|---|---|---|---|
| 5.1 |  3.5 |  1.4 |  0.2 | setosa| 
| 5.7	| 2.9 | 4.2 | 1.3 | versicolor | 
| 7.7 | 3.0  | 6.1  | 2.3  |  virginica |

We could use a full dataset with data like above to make a prediction of the flower species given only the Petal and Sepal Lengths.


Another good example of supervised learning is a email spam filter. 

Say we have a bunch of emails in our dataset and they all have a label of either `spam` or `not_spam`. We could then train a supervised learning model to look at all of those emails and pick up patterns that show up in the spam emails. There are probably certain words or formattiing that repeat them selves. If you've ever looked in your email spam folder you can probably pick out some of those things yourself!

There are 2 main types of supervised learning:

### Classification

Classification tries to assign the correct label to a new piece of data not containing a label. Both examples above are good examples of classfication problems.

Spam filter would look at an email and decide if it should be labeled as `spam` or `not_spam`

We could be given a new flower measurment and we want to try to label it with the correct Species: `setosa`, `versicolor`, `virginica`

| sepal length (cm)  | sepal width (cm)  | petal length (cm)  | petal width (cm)  |
|---|---|---|---|---|---|
| 5.1 |  3.5 |  1.4 |  0.2 |  

According to a model I trained it thinks this would be `versicolor`. 


### Regression

Regression tries to prdeict a target value

stock price 

Housing price


## Unsupervised Learning

Unsupervised Learning uses a dataset that is not labeled. 

### Clustering 

### Association


## Reinforcment Learning

Example

## Semi-Supervised Machine Learning

Example

## Deep Learning & Neural Networks

Seperate topic Want to do a quick mention here. Let me know if you'd like to see a workshop on deep learning basics!

Example



For this class we're going to stay focused on Supervised Machine learning. 

But what would you like to see a class on next?


## Supervised Learning Models

Some of the common models. Having an idea of what these do and applications they should be used for is important! 

[Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)

[Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)

[K-Nearest Neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

[Support Vector Machines](https://en.wikipedia.org/wiki/Support-vector_machine)

[Random Forest](https://en.wikipedia.org/wiki/Random_forest)


# Some Basics Terms:

We can only scratch the surface of Machine Learning tonight in this workshop, so this is by no means everything you need to know, but it should help you get started!

fitting / training

### Bias:

Examples

##### overfitting

##### underfitting

cross validation

XGboost (I heard this term all the time when I was first starting out)

loss function

Gradient Decent


# Machine Learning with Python:

## Popular Python Data & Machine Learning Libraries

Again this just some of them, there are soooooo many.....



Pandas

Numpy

matplotlib

Scikitlearn

Tensorflow

Pytorch

Keras

NLTK

OpenCV



*note* about anaconda




## Regression Project

Boston House price

Look at data. 

Look at expected outcome.

What model is good for this?

Prediction outcome: Classification or Regression?


### Linear Regression

Boston House price


## Classification Project

### k-NN Project


# YOU MADE IT THORUGH!

Did you learn something new?

Do you feel more comfortable with the ideas of Machine Learning?

Do you have an awesome idea you want to use try using machine learning? What is it?


## Recap

<details>
  <summary>What is a conditional in Python?</summary>
  
 A way to check if data meets a certain condition or not. `if` `elif` `else`.
	
</details>


## KEEP LEARNING!

Best way to learn is solving a problem you're excited about!

#### Resources
Siraj



## Upcoming events

What would you like to see next?










