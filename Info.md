# Emotion Recognition Application, Using Voice Recognition Model Built with Neural Networks

#### Graduation project by: 
  Eren Ali Aslangiray
  Mehmet Enis İşgören
  Sümeyye Sena Eminmollaoğlu
  Meryem Şahin
  
## Abstract:  
  This is our Graduation Project in İstanbul Şehir University CS department. Project deadline is in June 2019. The main goal of this project is, with using unique data collecting techniques we come up with an emotion recognition model that will take human speech as input and emotion status as an output. System will be also developed as a virtual assistant-like interaction with user so it will try to manipulate the user's current emotion stage while collecting the valuable human behaviour data in its back stage.

## Aim of the project:

  As an individual, we all want someone or something that understands us and see what we are going through or how we feel. From the start of the civilization we humans come together to overcome the problems we have and socialize to make every individual’s life better. From that perspective, our aim arises.
  The mood of a person may get affected from various things such as weather, time of the day or even the day of the week, likewise life. It keeps on going. We are willing to give a better daily life for individuals to have a good impact on their mood. Our design aims to predict users’ emotions and their response to certain events with the help of previously recorded reactions of users. Assistant will have a base case for general behaviors and will keep adapting from user’s preferences.
  
## Introduction:

  Voice is the most common communication tool used to interact with one another. Today, voice recognition applications are available for limited areas, but these areas are increasing steadily. In these applications, the machine detects the sound and gives appropriate answer to it. When communicating with phone, people prefer the sound rather than using keyboard, because it is easy and fast. We designed an application which will improve the currently used assistants such as Siri or Alexa, and with the help of this we will have an assistant which will develop empathy with the user and this system will be able to make mutual conversations. The application is going to ask some general information about users to learn what they like or dislike and their behaviors in certain situations at the beginning. Then it will get users’ voice and understand the meaning of the words and sentences in general. Not just the meaning, but also emotions will be recognized by the application from the voice. So that it will evaluate mood of the users at that time and return with some recommendations such as a movie, outside activity, food, product advertisement etc. Then Assistant will ask if he/she is satisfied from that recommendation and according to the answer, it will update its decision tree and adapt to upcoming user preferences.
  In a nutshell, our goal is to improve people’s daily lives by learning their moods and thoughts at any time of the day. So that we can make suggestions to make them feel better and to make them evaluate their free times. By this motivation, we foresee that our application will make people feel less lonely.

## Literature Review:

  There are many studies in the area of speech recognition and there are basic approaches that used in this problem. As discussed Gevaert, Tsenov and Mladenov in their paper, general structure of a speech recognition consists five steps which are; speech, signal processing, feature extraction, speech classification and output. In addition, there are commonly used techniques to achieve this problem. Dynamic Time Warping (DTW) compares words with reference words, Hidden Markov Modelling (HMM) splits the speech into small entities and it compares with the best-suited model. Another technique is Neural Networks which we will also be using in our project are similar to HMM, but Neural Networks use connection strengths instead of probabilities for state transitions (Gevaert, Tsenov & Mladenov, 2010, p.2). The article is mainly focusing on Neural Networks so it is very useful for our project.
  After the process of getting voice from the user and converting it to the text, with Natural Language Processing the meaning will be extracted from the text. There are 2 methods to do this which are Natural Language Processing for Speech Synthesis and Natural Language Processing for Speech Recognition. NLP for Speech Synthesis is based on text to speech conversion and it uses the sentence segmentation which deals with punctuation marks with a simple decision tree (Reshamwala, Mishra & Pawar, 2013, p.113). On the other hand, NLP for Speech Recognition is based on the grammar of a language (Trilla, 2009, p.3). It is needed to use Natural Language Processing in our project in terms of understanding user’s speech and return with some valuable suggestions.
  Decision trees are a decision support tool for regression or classification models. According to Jordan, Ghahramani and Saul in their research, we have to know probabilistic decision trees and Hidden Markov models to understand Hidden Markov decision trees. In probabilistic decision tree, decisions are modeled probabilistically and recursion spreading upward and downward in the tree. In Hidden Markov model, the key calculation fit in it, recursion extending forward or backward in the chain. Hidden Markov decision trees is a probabilistic decision tree (upward and downward) with Hidden Markov model (forward or backward).

## Project Pipeline
![alt text]()
