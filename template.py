def get_templates(dataset):
    if "celeb" in dataset:
        imagenet_templates = celeb_imagenet_templates
        base_imagenet_templates = celeb_base_imagenet_templates
    elif "animal" in dataset or "af" in dataset:
        imagenet_templates = animal_imagenet_templates
        base_imagenet_templates = animal_base_imagenet_templates
    elif "food" in dataset:
        imagenet_templates = food_imagenet_templates
        base_imagenet_templates = food_base_imagenet_templates
    elif "landscape" in dataset:
        imagenet_templates = lhq_imagenet_templates
        base_imagenet_templates = lhq_base_imagenet_templates
    else:
        print("@@@@@@@@@@@@@@@@ template should be registered in ./template.py for current dataset @@@@@@@@@@@@@@@@@@")
        exit(0)
    return imagenet_templates, base_imagenet_templates

celeb_imagenet_templates = [
    'a face photo with {}.',
    'a face photo of the {}.',
    'the face photo of the {}.',
    'a good face photo of the {}.',
    "high quality face photo of {}.",
    "a face image of {}.",
    "the face image of {}.",
    "high quality face image of {}.",
    "a high quality face image of {}.",
]

celeb_base_imagenet_templates = [
    ['a face photo with.'],
    ['a face photo of the.'],
    ['the face photo of the.'],
    ['a good face photo of the.'],
    ["high quality face photo of."],
    ["a face image of."],
    ["the face image of."],
    ["high quality face image of."],
    ["a high quality face image of."],
]


animal_imagenet_templates = [
    'a animalface photo with {}.',
    'a animalface photo of the {}.',
    'the animalface photo of the {}.',
    'a good animalface photo of the {}.',
    "high quality animalface photo of {}.",
    "a animalface image of {}.",
    "the animalface image of {}.",
    "high quality animalface image of {}.",
    "a high quality animalface image of {}.",
]

animal_base_imagenet_templates = [
    ['a animal photo with.'],
    ['a animal photo of the.'],
    ['the animal photo of the.'],
    ['a good animal photo of the.'],
    ["high quality animal photo of."],
    ["a animal image of."],
    ["the animal image of."],
    ["high quality animal image of."],
    ["a high quality animal image of."],
]


food_imagenet_templates = [
    'a food photo with {}.',
    'a food photo of the {}.',
    'the food photo of the {}.',
    'a good food photo of the {}.',
    "high quality food photo of {}.",
    "a food image of {}.",
    "the food image of {}.",
    "high quality food image of {}.",
    "a high quality food image of {}.",
]

food_base_imagenet_templates = [
    'a food photo with {}.',
    'a food photo of the {}.',
    'the food photo of the {}.',
    'a good food photo of the {}.',
    "high quality food photo of {}.",
    "a food image of {}.",
    "the food image of {}.",
    "high quality food image of {}.",
    "a high quality food image of {}.",
]

lhq_imagenet_templates = [
    'a scene photo with {}.',
    'a scene photo of the {}.',
    'the scene photo of the {}.',
    'a good scene photo of the {}.',
    "high quality scene photo of {}.",
    "a scene image of {}.",
    "the scene image of {}.",
    "high quality scene image of {}.",
    "a high quality scene image of {}.",
]

lhq_base_imagenet_templates = [
    ['a scene photo with.'],
    ['a scene photo of the.'],
    ['the scene photo of the.'],
    ['a good scene photo of the.'],
    ["high quality scene photo of."],
    ["a scene image of."],
    ["the scene image of."],
    ["high quality scene image of."],
    ["a high quality scene image of."],
]