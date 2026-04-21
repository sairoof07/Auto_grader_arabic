import streamlit as st 
# VLM 
from ollama import chat 
# my funcitons 
from Funcs_Vars import preprocess, fix_typos
import cv2
import ast
import tempfile, os
import numpy as np
import pandas as pd 
import json 
import matplotlib.pyplot as plt
@st.cache_resource
def load_nltk():
    import nltk
    nltk.download('stopwords')
    return nltk.corpus.stopwords.words('arabic')
@st.cache_resource
def load_model():
    return SentenceTransformer("models/mpnet-base-all-nli-triplet/final")
#NLP
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer 
from sentence_transformers import util
arabic_stops = load_nltk()
model = load_model()
model_ans = []
st.header('Automatic Paper grading app')
col1, col2, col3= st.columns(3, vertical_alignment="bottom")
st.sidebar.image('logo.png')
VLM = st.sidebar.selectbox(
"which model do you want to use?",
("qwen2.5vl:7b" , "qwen3-vl:8b", "qwen3-vl:30b" ))
alpha = st.sidebar.number_input('how strict you want the model? (0 is not strict, 1 is very strict) ' , min_value = 0.0 , max_value = 1.0, value = 0.7)
st.sidebar.write('powered by Ollama QWEN')

st.write(' ')
with col1:
    tests = st.file_uploader('Upload the test here', type = ["jpg", "jpeg", "png", 'pdf'] , accept_multiple_files = True)    
with col2:
    n= st.number_input('enter the number of questions', value= 1, step=1 )
st.write('make sure that all papers have the same questions' , )
for i in range(n):
    model_ans.append(st.text_input(f'model answer for question number {i+1}', key = i))
    i +=1 
keys = [f"Q{i}" for i in range(1, n+1)]
model_answer = dict(zip(keys, model_ans)) 
with col3: 
    submit = st.button('submit')
if not submit :
    st.stop()
prompt = f"""You are a strict data extraction tool analyzing an Arabic exam image.
Task: Extract exactly {n} questions.
Output: STRICT valid JSON only. No conversational text. No markdown backticks.
### RULES
1. Return exactly {n} questions, ordered by question_number ascending (1..{n}).
2. question_type: 1 if the question has labeled choices (أ, ب, ج, د) to pick from and only include the letter in the file; 2 if true/false (صح/خطأ); 0 ONLY for free-written responses. Read the question before choosing — do not default.
3. grade_position: {{"x": int, "y": int}} pixel coords immediately to the LEFT of the question number.
4. student_answer: what the student wrote. If not detected, return "__".
5. Always return one JSON object with a single top-level key "questions" (a list of length {n}).
### EXACT OUTPUT SCHEMA
{{
  "questions": [
    {{
      "question_number": int,
      "question_type": int,
      "question_text": "string",
      "student_answer": "string",
      "grade_position": {{"x": int, "y": int}}
    }}
  ]
}}"""
model_ans = pd.Series(list(model_answer.values()))
if 'results' not in st.session_state:
    st.session_state.results = []
if submit:
    st.session_state.results = []
if not st.session_state.results:
    with st.status("Grading exams...", expanded=True) as status:
        for i in tests:
            st.write(f"📄 Processing: {i.name}")
            # Save upload to temp file
            suffix = os.path.splitext(i.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(i.read())
                tmp_path = tmp.name
            # Load original image
            original_img = cv2.imread(tmp_path)
            h, w = original_img.shape[:2]
            scale_w = w / 800
            resized = cv2.resize(original_img, (800, int(h * 800 / w)))
            cv2.imwrite(tmp_path, resized)
            
            st.write("🤖 Running VLM..." , VLM )
            response = chat(
                model= VLM, 
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [tmp_path]
                }]
            )
            
            st.write("🔍 Parsing response...")
            raw = response['message']['content']
            raw = raw.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
            data = json.loads(raw)
            if isinstance(data, list):
                data = {'questions': data}
            st.write("model_ans:", model_ans.tolist())
            dataframe = pd.DataFrame(data['questions'])
            dataframe['grade'] = 0
            dataframe['model_answer'] = model_ans
            st.write("df after assign:", dataframe)
            st.write("question_types:", dataframe['question_type'].tolist())
            mask = dataframe['question_type'] == 0
            dataframe['student_answer'] = dataframe.apply(
                lambda row: fix_typos(row['student_answer'], row['model_answer']), axis=1)
            def _safe_pp(t):
                out = preprocess(t)
                return out if out else t
            dataframe.loc[mask, 'student_answer'] = dataframe.loc[mask, 'student_answer'].apply(_safe_pp)
            dataframe.loc[mask, 'model_answer'] = dataframe.loc[mask, 'model_answer'].apply(_safe_pp)
            st.write("📐 Computing similarity...")
            dataframe.loc[mask, 'student_vec'] = dataframe.loc[mask, 'student_answer'].apply(model.encode)
            dataframe.loc[mask, 'model_vec'] = dataframe.loc[mask, 'model_answer'].apply(model.encode)
            dataframe.loc[mask, 'similarity'] = [
                util.cos_sim(s, m).item()
                for s, m in zip(dataframe.loc[mask, 'student_vec'], dataframe.loc[mask, 'model_vec'])
            ]
            dataframe.loc[
                (dataframe['question_type'].isin([1, 2])) &
                (dataframe['student_answer'] == dataframe['model_answer']), 'grade'] = 1
            dataframe.loc[(dataframe['question_type'] == 0) & (dataframe['similarity'] >= alpha), 'grade'] = 1
            st.write("🖊️ Drawing grades on image...")
            img_draw =  resized.copy()
            H, W = img_draw.shape[:2]
            n_rows = len(dataframe)
            for _, row in dataframe.iterrows():
                x = 30
                y = int(H * (int(row['question_number']) - 0.5) / n_rows)
                grade = int(row['grade'])
                color = (0, 180, 0) if grade == 1 else (0, 0, 200)
                label = '1' if grade == 1 else '0'
                cv2.circle(img_draw, (x, y), 18, color, -1)
                cv2.circle(img_draw, (x, y), 18, (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(img_draw, label, (x - tw // 2, y + th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            st.session_state.results.append({
                "name": i.name,
                "image": img_draw,
                "df": dataframe
            })
            os.unlink(tmp_path)
        status.update(label="✅ Done!", state="complete")
for res in st.session_state.results:
    with st.expander(f"📋 Graded: {res['name']}"):
        st.image(res['image'], channels="BGR")
        st.dataframe(res['df'][['question_number', 'question_text', 'student_answer', 'model_answer', 'grade']])
st.write('Project by: Ali Alsairafi')