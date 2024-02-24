#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:57:26 2022

@author: Amine Laghaout
"""

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


def sentiment(text="I've been waiting for this thing all my life."):
    classifier = pipeline("sentiment-analysis")
    res1 = classifier(text)
    print("res1:", res1)

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
    res2 = classifier(text)
    print("res2:", res2)

    return res1, res2


def generate(
    text="In this course, we will teach you how to",
    model="distilgpt2",
    max_length=30,
    num_return_sequences=2,
):
    generator = pipeline("text-generation", model=model)

    res = generator(
        text, max_length=max_length, num_return_sequences=num_return_sequences
    )

    print(res)

    return res


def classify(
    text="This is a course about quantum physics",
    candidate_labels=["education", "politics", "business"],
):
    classifier = pipeline("zero-shot-classification")

    res = classifier(text, candidate_labels=candidate_labels)

    print(res)

    return res


# %%
if __name__ == "__main__":
    res = sentiment()
    # res = generate("Do you know how I go to the moon?")
    # res = classify("A course on how to make money with quantum physics")

    print(res)
