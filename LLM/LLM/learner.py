#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:57:26 2022

@author: Amine Laghaout
"""

import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import torch
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
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
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


def image_segmentation():
    import requests
    from PIL import Image

    from transformers import AutoImageProcessor, DetrForSegmentation

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50-panoptic"
    )
    model = DetrForSegmentation.from_pretrained(
        "facebook/detr-resnet-50-panoptic"
    )

    # prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt")

    # forward pass
    outputs = model(**inputs)

    # Use the `post_process_panoptic_segmentation` method of the
    # `image_processor` to retrieve post-processed panoptic segmentation maps.
    # Segmentation results are returned as a list of dictionaries
    result = image_processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[(300, 500)]
    )

    # A tensor of shape (height, width) where each value denotes a segment id,
    # filled with -1 if no segment is found
    panoptic_seg = result[0]["segmentation"]

    # Get prediction score and segment_id to class_id mapping of each segment
    panoptic_segments_info = result[0]["segments_info"]

    return dict(
        result=result,
        panoptic_seg=panoptic_seg,
        panoptic_segments_info=panoptic_segments_info,
        outputs=outputs,
    )


def image_segmentation2():
    from transformers import AutoImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image
    import requests

    image = "D:/Pictures/2023/20230321_154545.JPG"
    if image[:5] == "https":
        image = Image.open(requests.get(image, stream=True).raw)
    else:
        image = Image.open(open(image, "rb"))

    image_processor = AutoImageProcessor.from_pretrained(
        "facebook/detr-resnet-50"
    )
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to Pascal VOC format
    # (xmin, ymin, xmax, ymax)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(
        outputs, threshold=0.9, target_sizes=target_sizes
    )[0]

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )


def image_recognition(images):
    from transformers import pipeline

    vision_classifier = pipeline(model="google/vit-base-patch16-224")
    preds = vision_classifier(images=images)
    preds = [
        {"score": round(pred["score"], 4), "label": pred["label"]}
        for pred in preds
    ]
    return preds, vision_classifier


def show_output_embeddings():
    # https: // huggingface.co/docs/transformers/main_classes/output
    text = [
        "This is awesome",
        "This sucks",
        "This is good",
        "I'm lukewarm about this",
    ]
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    inputs = tokenizer(text[0], return_tensors="pt")
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(
        **inputs,
        labels=labels,
        output_hidden_states=True,
        output_attentions=True,
    )
    print(outputs[:])

    return outputs


def show_tokenization(padding=False):
    text = ["Using a Transformer network is simple simple", "Simply put!"]

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    classifier = pipeline(
        "sentiment-analysis", model=model, tokenizer=tokenizer
    )
    res2 = classifier(text)
    print("res2:", res2)
    print("-----------")
    tokenized_sequence = tokenizer(text, padding=padding)
    print("- tokenizer(text):\n", tokenized_sequence)
    tokens = tokenizer.tokenize(text)
    print("- tokens = tokenizer.tokenize(text):\n", tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print("- ids = tokenizer.convert_tokens_to_ids(tokens):\n", ids)
    decoded_string = tokenizer.decode(ids)
    print("- decoded_string = tokenizer.decode(ids):\n", decoded_string)
    encoded_string = tokenizer.encode(text)
    print("- encoded_string = tokenizer.encode(text):\n", encoded_string)
    print(
        "- tokenizer.decode([454, 999, 3722, 2003, 10, 2121]):\n",
        tokenizer.decode([454, 999, 3722, 2003, 10, 2121]),
    )


def sentence_similarity():
    from transformers import AutoTokenizer, AutoModel
    import torch
    import torch.nn.functional as F

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    sentences = [
        "This is an example sentence",
        "Each sentence is converted",
        "Now what?",
        "Each sentence is diverted",
        "This is an example sentence",
    ]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(
        model_output, encoded_input["attention_mask"]
    )

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    print("Sentence embeddings:")
    # print(sentence_embeddings)

    return dict(
        sentence_embeddings=sentence_embeddings, model_output=model_output
    )


# %% __main__
if __name__ == "__main__":
    pass
    # Sentiment
    # res = sentiment()
    # print(res)

    # Generate text
    # res = generate("Do you know how I go to the moon?")
    # print(res)

    # Classify text
    # res = classify(
    #     ["A course on how to make money with quantum physics.",
    #      "The stocks have been going up lately.",
    #      "The runner-up in the election is Jack Leighton"])
    # print(res)

    # Image recognition
    # preds, vision_classifier = image_recognition(
    #     'https://b-cdn.springnest.com/media/img/9u/102b15bc.jpg'
    # )

    similarity = sentence_similarity()
    v = similarity["sentence_embeddings"]
    print(v[0] @ v[4])
