This repository contains small library for creating reccomender systems. <br>
For some time now I've been interested in reccomender systems algorithms. Almost all apps like Netflix or Goodreads use reccomendations for user as one of the main features.So why is that there are none usable frameworks/libraries for them? I always thought that reccomender systems are overeaten topic. And I was very wrong.

## What is a reccomender system 
![meme1](https://scontent.fpoz4-1.fna.fbcdn.net/v/t1.6435-9/176724894_2848797975437436_6717594420218544508_n.jpg?_nc_cat=1&ccb=1-3&_nc_sid=8bfeb9&_nc_ohc=fFaIniSHa98AX_LNxkX&_nc_ht=scontent.fpoz4-1.fna&oh=f4edac4c6b7c80a6de8ccdb1454778c9&oe=60A52DAD)
A recommender system, or a recommendation system (sometimes replacing 'system' with a synonym such as platform or engine), is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. Recommender systems are used in a variety of areas, with commonly recognised examples taking the form of playlist generators for video and music services, product recommenders for online stores, or content recommenders for social media platforms and open web content recommenders. These systems can operate using a single input, like music, or multiple inputs within and across platforms like news, books, and search queries. There are also popular recommender systems for specific topics like restaurants and online dating. Recommender systems have also been developed to explore research articles and experts, collaborators, and financial services. 

## How it works
There are three main ways of creating a reccomender system - content based, using collaborative filtering, and in the end a hybrid that connects both.<br>
![ex1](https://miro.medium.com/max/2400/1*aSq9viZGEYiWwL9uJ3Recw.png)
First one can be based on some clustering algorithm, that will grouup all users into several clusters based on what items where they using.
![ex2](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/67630/versions/6/screenshot.png)
Collaborative filltering is a little bit more tricky approach. Let's assume we have a dataset of movies and their ratings from different users. We can express it as
a matrix of shape (number_of_movies, number_of_users), where each row represents a movvie, and one column user ratings. That will be our target matrix. Also we know that each movie is represented by some genres (X vector), and each user like this genres to some extent (Theta vector). Thus, a dot product of X and Theta will give us predictions of users ratings on different movies. We can use this ratings to reccomend ones that our algorithm predicted to suit given user best. This approach although requiers from us to know beforehand X and Theta vector. Or do it? Thanks to fancy machine learning optimization algorithms like gradient descent or adam we can train these vectors based only on target Matrix.
![ex3](https://developers.google.com/machine-learning/recommendation/images/1Dmatrix.svg)
In the end we have hybrid approach - which is probably used by all big companies like FANG. It combiens cluster analysis with predictions from collaborative filtering algorithm to give most accurate predictions of reccomendations for the user.

## TODO:
 - Readme
 - Minibatches
 - Different Optimizers
 - Deep Matrix Factorization
 - Clustering reccomender
 - Hybrid reccomender
