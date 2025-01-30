

def gentypes():
    from transformers import pipeline
    unmasker = pipeline('fill-mask', model='bert-base-cased')
    print(unmasker("My friend, [MASK], excels in the field of mechanical engineering."))

    print(unmasker("My friend, [MASK], makes a wonderful kindergarten teacher."))

gentypes()