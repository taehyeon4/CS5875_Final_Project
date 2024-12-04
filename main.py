from src import cvmodule, nlpmodule

# computer vision
cvinstance = cvmodule.CVModule()
curr_sentence = cvinstance.run()
print(f"Raw text: {curr_sentence}")

# to override the computer vision part
#curr_sentence = "i hate this shit, i'm a dumbass, i'm stupid as hell"

# language processing
nlp_instance = nlpmodule.NLPModule()
processed_string = nlp_instance.process_text(curr_sentence)
print(f"Processed text: {processed_string}")