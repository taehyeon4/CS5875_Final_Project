from src import nlpmodule, cvmodule

# computer vision part
# cvinstance = cvmodule()

#temp final string
final_string = "my name is akhil and i fucking hate this pizza it tastes so ass"

#language processing part
nlp_instance = nlpmodule.NLPModule()
processed_string = nlp_instance.process_text(final_string)
print(f"Processed text: {processed_string}")