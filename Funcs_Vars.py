from transformers import pipeline

import nltk
nltk.download('stopwords')
arabic_stops = nltk.corpus.stopwords.words('arabic')
#general tools
import cv2 
import pandas as pd 
import json 
import matplotlib.pyplot as plt
#preprocessing 
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.normalize import normalize_alef_ar, normalize_unicode
from camel_tools.utils.dediac import dediac_ar

#NLP
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer 
from sentence_transformers import util
def fix_typos(student_text, model_text):
    model_words = model_text.split()
    fixed = []
    for word in student_text.split():
        match, score, _ = process.extractOne(word, model_words, scorer=fuzz.ratio)
        if score >= 80:
            fixed.append(match)
        else:
            fixed.append(word)
    return ' '.join(fixed)


def preprocess(text):
    text = normalize_unicode(text)
    text = normalize_alef_ar(text)
    text = dediac_ar(text)
    text = text.replace('ـ', '')
    text = ' '.join([word for word in text.split() if word not in arabic_stops])
    return text.strip()

def fix_typos(student_text, model_text):
    model_words = model_text.split()
    fixed = []
    for word in student_text.split():
        match, score, _ = process.extractOne(word, model_words, scorer=fuzz.ratio)
        if score >= 80:
            fixed.append(match)
        else:
            fixed.append(word)
    return ' '.join(fixed)
#n=0
#prompt = f"""This is an Arabic exam paper with exactly {n} questions. Find ALL {n} questions.
#
#The correct answer for each question is:
#{chr(10).join([f"Q{i+1}: {ans}" for i, ans in enumerate(model_ans)])}
#
#For each question, set "model_answer" to its corresponding answer from the list above.
#
#Return a JSON object:
#{{
#  "questions": [
#    {{
#      "question_number": 1,
#      "question_text": "question text in Arabic",
#      "question_type": 0,
#      "student_answer": "what the student wrote or selected, empty string if blank",
#      "model_answer": "the correct answer for this question from the list above",
#      "grade_position": {{"x": 0, "y": 0}}
#    }}
#  ]
#}}
#
#question_type: 0 = written, 1 = multiple choice, 2 = correct/wrong, default to 0 if not type 1 or 2
#student_answer: for multiple choice, write the letter the student circled/marked. For written, write what they wrote. Empty string if nothing.
#grade_position: pixel coordinates to the LEFT of the question number
#Return ONLY the JSON. No extra text.
#"""
#prompt = f"""Arabic exam image. This exam has {n} questions. Find ALL {n} questions, do not skip any. Return ONLY this JSON, no extra text:
#{{"questions":[{{"question_number":1,"question_type":0,"student_answer":"","model_answer":"","grade_position":{{"x":0,"y":0}}}}]}}
#
#question_type: 0=written, 1=multiple choice, 2=correct/wrong, default to 0
#grade_position: pixels, place LEFT of question number, must be a JSON object with integer x and y keys
#model_answer: from this dict: {model_answer}"""
