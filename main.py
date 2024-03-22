import json
import os
from fastapi import FastAPI
from moviepy.audio.AudioClip import CompositeAudioClip
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
import torch
from TTS.api import TTS
from moviepy.editor import ImageClip, AudioFileClip, VideoFileClip, concatenate_videoclips
import requests
from PIL import Image, ImageDraw, ImageFont
from fastapi.responses import FileResponse
from datetime import datetime
import sqlite3
import os

DATABASE_URL = "apicalls.db"
con = sqlite3.connect(DATABASE_URL)

def save_api_call_to_db(subject, mood, style):
    con = sqlite3.connect("DATABASE_URL")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS apiCalls(id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, mood TEXT, style TEXT, createdAt TEXT)")
    cur.execute("""INSERT INTO apiCalls (subject, mood, style, createdAt) VALUES (?, ?, ?, ?)""", (subject, mood, style, datetime.now()))
    con.commit()

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

model = "gpt-4-0125-preview"

class StoryBody(BaseModel):
    subject: str
    mood: str
    style: str
    openaiApiKey: str

app = FastAPI()

@app.post("/create")
async def create_story_video(story_body: StoryBody):
    client = OpenAI(api_key=story_body.openaiApiKey)
    save_api_call_to_db(subject=story_body.subject, mood=story_body.mood, style=story_body.style)        
    story = await create_story(story_body, client)
    background_music_id = get_background_music_id(f"A {story_body.mood} story about ${story_body.subject}", client)
    create_video(story, background_music_id, story_body.style, story_body.subject, client)
    return FileResponse("merged_videos.mp4", media_type='video/mp4', filename='merged_videos.mp4')


async def create_story(story_body, client):
    user_prompt = f"Please write a {story_body.mood} story about ${story_body.subject}"
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a story teller. You write 3 sentence long stories. the sentences shouldn't contain '.' except the end of a sentence"},
            {"role": "user", "content": user_prompt}
        ])
    story = completion.choices[0].message.content
    return story


def get_background_music_id(subject, client):
    similarities = {}
    with open("./music/vectors.json", "r") as file:
        vectors = json.load(file)
    with open("./music/index.json", "r") as file:
        index = json.load(file)

    subject_vector = get_vector(subject, client)
    for description, vector in vectors.items():
        similarities[description] = cosine_similarity(vector, subject_vector)
    similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return index[similarities[0][0]]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_vector(text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def create_video(story, background_music_id, style, subject, client):
    story_sentences = story.split(".")
    count = 0
    for sentence in story_sentences:
        try:
            print(sentence)
            sentence_file_path = f"sentences/sentence{count}.wav"
            image_file_path = f"images/image{count}.png"
            clip_file_path = f"clips/clip{count}.mp4"
            create_image(sentence, image_file_path, style, subject, client)
            create_video_with_speech(sentence, image_file_path, sentence_file_path, clip_file_path)
            count = count + 1
            print("done")
        except:
            continue

    video_files = [filename for filename in os.listdir("clips") if filename.endswith(".mp4")]
    video_files.sort()
    video_clips = [VideoFileClip(os.path.join("clips", video_file)) for video_file in video_files]
    concatenated_clip = concatenate_videoclips(video_clips, method='compose')

    add_background_music(concatenated_clip, background_music_id)

    concatenated_clip.write_videofile("merged_videos.mp4", codec='libx264', audio_codec='aac', fps=24)


def create_image(sentence, image_path, style, context, client):
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"style: {style}; sentence: {sentence}; context:{context}",
        size="1024x1024",
        quality="standard",
        n=1,
    )

    image_url = response.data[0].url
    response = requests.get(image_url)
    with open(image_path, 'wb') as file:
        file.write(response.content)

    # add_subtitle_to_image(image_path, sentence, image_path)


def create_video_with_speech(sentence, image_file_path, sentence_file_path, clip_file_path):
    tts.tts_to_file(text=sentence, speaker_wav="speaker.wav", language="en", file_path=sentence_file_path)
    sentence_audio = AudioFileClip(sentence_file_path)
    with ImageClip(image_file_path) as video_clip:
        video_clip = video_clip.set_duration(sentence_audio.duration)
        final_clip = video_clip.set_audio(sentence_audio)
        final_clip.write_videofile(clip_file_path, codec='libx264', audio_codec='aac', fps=24)


def add_subtitle_to_image(image_path, subtitle_text, output_path, position=(10, 10), font_size=24, font_color="white"):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)
    draw.text(position, subtitle_text, font=font, fill=font_color)
    image.save(output_path)


def add_background_music(concatenated_clip, background_music_id):
    background_music = AudioFileClip(f"music/{background_music_id}.mp3")
    background_music = background_music.set_duration(concatenated_clip.duration)
    original_audio = concatenated_clip.audio
    composite_audio = CompositeAudioClip(
        [original_audio, background_music.set_start(0).volumex(0.5)])
    concatenated_clip.audio = composite_audio

def add_subtitle_to_image(image_path,sentence):
    i = Image.open(image_path)
    Im = ImageDraw.Draw(i)
    mf = ImageFont.truetype("arial.ttf", 45)


    text = sentence.split()
    n = 5
    subtitles='\n '.join([' '.join(text[i:i+n]) for i in range(0,len(text),n)])


    Im.text((5,i.height*0.8), subtitles, (255,255,255), font=mf)


    i.save(image_path)
