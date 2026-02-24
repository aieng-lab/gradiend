"""
Minimal start-to-end GRADIEND workflow with TextPredictionDataCreator.

Uses 75+ artificial sentences (3SG: he/she/it, 3PL: they, neutral).
Matches docs/start.md. Run: python -m gradiend.examples.start_workflow
"""

from gradiend import (
    TextFilterConfig,
    TextPredictionDataCreator,
    TrainingArguments,
    TextPredictionTrainer,
)

# 75+ sentences: mix of 3SG (he/she/it), 3PL (they), both in one sentence, and neutral.
ARTIFICIAL_TEXTS = [
    "The chef tasted the soup, then he added a pinch of pepper and stirred it.",
    "The pianist closed her eyes and played the final chord; she had practised it for weeks.",
    "The dog ran to the door; it wanted to go outside and chase the ball.",
    "The report will be ready for review by the end of the week.",
    "His phone rang twice before he picked it up and left the room.",
    "She handed the package to the courier and asked them to deliver it by noon.",
    "The players on both teams huddled on the pitch before they ran back to their positions.",
    "The author read a paragraph from his novel to the audience and then they asked him questions.",
    "Breakfast is included for all guests staying at the hotel.",
    "The nurse checked her clipboard and told the family members they could all visit soon.",
    "The birds gathered on the wire; when the cat moved they flew away at once.",
    "The mechanic wiped his hands and said the car would be ready by Friday; he had fixed it.",
    "She opened the window and watched the leaves fall in the garden below.",
    "The committee members met on Tuesday and they voted to postpone the decision.",
    "When the referee blew the whistle he showed the player a yellow card.",
    "The volunteers packed the boxes and said they would all load the van at dawn.",
    "Her brother and his colleagues sent a text saying they were stuck in traffic together.",
    "The dog barked at the postman and he dropped the letters when it lunged.",
    "The meeting ran over time so the chair cut the last two items; the participants agreed they would meet again.",
    "The weather improved by the afternoon and the streets dried quickly.",
    "Rain poured all morning but the basement stayed dry; the pump had run all night and it held.",
    "She found the recipe in the drawer; it calls for butter, flour, and a pinch of salt.",
    "The road was closed for repairs and the local officials said they expected the detour would add twenty minutes.",
    "The film ended at midnight and the group of friends all went home; they shared umbrellas in the rain.",
    "The museum opens at nine and closes at six on weekdays.",
    "Trains run every fifteen minutes during peak hours.",
    "The recipe works best when the oven is preheated properly.",
    "The meeting has been moved to the smaller room on the third floor.",
    "The gardener pruned the roses and he left the cuttings by the gate.",
    "The drummer lost her stick in the middle of the song but she kept the beat.",
    "One of the students raised her hand and asked them to repeat the question.",
    "The board members announced that they had approved the budget; the CEO signed it.",
    "He left the book on the table and she noticed the door was open when it swung.",
    "The car broke down on the motorway and it had to be towed away.",
    "The cat jumped off the wall; it landed in the flower bed and ran off.",
    "Two colleagues from his team offered him a seat but he preferred to stand; she nodded and they sat down.",
    "Coffee was served in the lobby while the conference continued upstairs.",
    "Keys were left on the counter by the front door.",
    "Parking is available in the lot behind the building.",
    "The driver signalled left and he turned into the car park.",
    "She replied to the email before the other team members had a chance to follow up, and they thanked her later.",
    "The laptop was slow so it needed a restart and more memory.",
    "The lecture will be recorded and posted online by tomorrow.",
    "His keys fell under the seat and he had to reach for them.",
    "The team members celebrated after they won the final match.",
    "The manager gave his approval and then the project team met; they scheduled the launch.",
    "They invited her to the meeting and she accepted on the spot.",
    "Lunch will be served in the canteen from twelve to two.",
    "The doctor checked her notes and told the patients they could go home; they thanked her at the door.",
    "The fans in the crowd cheered when they saw the result on the screen.",
    "When the alarm went off he switched it off and got up.",
    "The staff members finished the inventory and they reported the count.",
    "Her colleague forwarded the file and the two of them opened it together; they checked every page.",
    "The cat stretched and it jumped onto the sofa.",
    "The council members met last night and they approved the new bylaws.",
    "The coach gave his feedback and the players practised the drill again; they repeated it three times.",
    "The intern made her first presentation and her colleagues in the room asked a few questions; they praised her work.",
    "The dog waited by the bowl; it had not been fed yet.",
    "The panel members discussed the proposal and they reached a consensus.",
    "He locked the office and she set the alarm before they left the building together.",
    "The van pulled up and it unloaded the delivery at the back.",
    "The results are published on the intranet every Friday.",
    "The neighbours across the street waved to her and she waved back as they all passed by.",
    "Snow fell all day but the gritters had been out and it cleared.",
    "The schedule is on the wall next to the break room.",
    "The bus was late so her friends and she missed the start of the film; they had to sneak into their seats.",
    "The document is in the shared folder and can be edited by anyone.",
    "The waiter brought the bill and he left the tip on the table.",
    "The singer forgot the words but she carried on and the audience members applauded; they gave her a standing ovation.",
    "The printer ran out of paper and it stopped mid-job.",
    "The deadline has been extended to the end of the month.",
    "The committee will reconvene next week to finalise the report.",
    "Tea and biscuits are provided in the kitchen on each floor.",
    "She booked the room and the hotel staff confirmed the reservation by email; they also sent her directions.",
    "The gate was left open so the horse got out and it wandered off.",
    "The contract is valid for twelve months from the signing date.",
]

NEUTRAL_EXCLUDE = [
    "i", "we", "you", "he", "she", "it", "they",
    "me", "us", "him", "her", "them",
]


def create_data():
    """Build training and neutral data with TextPredictionDataCreator (matches start.md)."""
    creator = TextPredictionDataCreator(
        base_data=ARTIFICIAL_TEXTS,
        feature_targets=[
            TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
            TextFilterConfig(targets=["they"], id="3PL"),
        ],
    )
    training = creator.generate_training_data(max_size_per_class=50)
    neutral = creator.generate_neutral_data(
        additional_excluded_words=NEUTRAL_EXCLUDE,
        max_size=50,
    )
    return training, neutral


def train_and_evaluate(training, neutral):
    """Train, evaluate encoder/decoder, and return the selected changed model (matches start.md)."""
    args = TrainingArguments(
        train_batch_size=4,
        eval_steps=5,
        num_train_epochs=5,
        max_steps=25,
        learning_rate=1e-3,
        experiment_dir='runs/examples/start_workflow'
    )
    trainer = TextPredictionTrainer(
        model="bert-base-uncased",
        data=training,
        eval_neutral_data=neutral,
        max_counterfactuals_per_sentence=2,
        img_format='png',
        args=args,
    )
    trainer.train()
    trainer.plot_training_convergence()

    enc_result = trainer.evaluate_encoder(plot=True)
    print("Correlation:", enc_result.get("correlation"))
    dec = trainer.evaluate_decoder(plot=True, target_class="3SG")
    changed_base_model = trainer.rewrite_base_model(decoder_results=dec, target_class="3SG")

    return trainer, enc_result, dec, changed_base_model


training, neutral = create_data()
trainer, enc_result, dec, changed_base_model = train_and_evaluate(training, neutral)
