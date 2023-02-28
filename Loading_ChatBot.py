
import json
import random
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from pathlib import Path, PurePath
from tkinter import *

path = Path.cwd()
lemmatizer = WordNetLemmatizer()
model = load_model(PurePath(path, "chatbot_model.h5"))
intents = json.loads(open(PurePath(path, "intents.json")).read())
words = pickle.load(open(PurePath(path, "words.pkl"),'rb'))
classes = pickle.load(open(PurePath(path, "classes.pkl"),'rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - splitting words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reducing to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for words that exist in sentence
def bag_of_words(sentence, words, show_details=True):
    # tokenizing patterns
    sentence_words = clean_up_sentence(sentence)
    # bag of words - vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,word in enumerate(words):
            if word == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % word)
    return(np.array(bag))

def predict_class(sentence):
    # filter below  threshold predictions
    p = bag_of_words(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sorting strength probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

#Creating tkinter GUI
def send(event=None):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != "":
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 10 )) #dialog box
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        ChatBox.insert(END, "Sam: " + res + '\n\n')
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
    if msg == "q":
        root.destroy()

root = Tk()
root.title("Scramjet Chatbox")
root.geometry("500x465")
root.resizable(width=True, height=True)
#Create Chat window
#ChatBox = Text("welcome to chatbox")
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial") #, wrap=CHAR)
ChatBox.config(state=DISABLED)

# the first sentence
ChatBox= Text()
ChatBox.config(foreground="#446665", font=("Verdana", 10 ))
ChatBox.pack()
ChatBox.insert("0.0", "Sam: Hi, how can I be of some help?\n\t Type 'q' for quit.\n\n")

#Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview,  orient='vertical') #cursor="heart",
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="10", height=3,
                    bd=0, bg="#ff8000", activebackground="#00294a",fg='#00294a',
                    command=send)

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")
EntryBox.pack()
EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=476,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=470)
EntryBox.place(x=6, y=401, height=50, width=365)
SendButton.place(x=376, y=401, height=50)
root.mainloop()

