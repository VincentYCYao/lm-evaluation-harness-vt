import json
import os
from typing import Union, List

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
# from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean


def topk_match_fn(gold, preds): 


    preds, gold = [pred.lower() for pred in preds], [target.lower() for target in gold]
    hits = 0
    for pred in preds:
        if pred in gold:
            hits += 1 
            break
        
    return hits


class BioLAMA_Wikidata(ConfigurableTask):
    DATASET_PATH = "JadeCheng/Biolama-wikidata"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0 # "Yaml"
    DATASET_NAME = None
    # CONFIG = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})

    def doc_to_text(self, doc):
        # Wikidata IDs and Prompts
        pid2prompt_meta = {"P2176": {"template": "The standard treatment for patients with [X] is a drug such as [Y]."},
                           "P2175": {"template": "[X] has effects on diseases such as [Y]."},
                           "P4044": {"template": "[X] cures diseases such as [Y]."},
                           "P780": {"template": "[X] has symptoms such as [Y]."},
                           "P2293": {"template": "Gene [X] has a genetic association with diseases such as [Y]."}}
        
        # Make the template for that specific doc.
        template =  pid2prompt_meta[doc["predicate_id"]]["template"]

        subject = doc["sub_label"]
        sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

        prefix = f'Consider the following sentence: "{sentence}"'
        suffix = '\n\n-> Which noun-phrase should <BLANK> be filled with? Give me 5 most probable candidates. Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", where the value for key "top_1" is the most promising entity that would replace <BLANK>.'

        prompt = prefix + suffix
        return f"{prompt}\n"

    def doc_to_target(self, doc):
        objects = []
        if 'obj_labels' in doc:
            objects = doc['obj_labels']  
        elif 'obj_label' in doc:
            objects = [doc['obj_label']]

        if 'obj_aliases' in doc:
            objects += [a for al in doc['obj_aliases'] for a in al]

        lower_objects = list(dict.fromkeys([obj.lower() for obj in objects]))

        # print(f"lower_objects = {lower_objects}")

        return lower_objects

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    # def validation_docs(self):
    #     # print(f"self.dataset = {self.dataset}")
    #     return self.dataset["validation"]
    
    def test_docs(self):
        return self.dataset["test"]

    def process_results(self, doc, results): 

        gold = self.doc_to_target(doc) # lower_objects

        def dict_to_list(topk_dict, k=5):

            ordered_list = []

            # Loop through keys from top_1 to top_k
            for i in range(1, k+1):
                key = "top_" + str(i)
                if key in topk_dict:
                    ordered_list.append(topk_dict[key])
                        
            return ordered_list


        def apply_to_resp(resp): 

            try:
                topk_dicts = json.loads(resp[0])
                # print(topk_dicts)
                # print("The first string is valid JSON.")
            except json.decoder.JSONDecodeError:
                # Handle the specific case where the JSON is invalid
                print("Invalid JSON encountered.")
                topk_dicts = {'top_1': '', 'top_2': '', 'top_3': '', 'top_4': '', 'top_5': ''}
                # return None  # or return {}, [] depending on what you expect
            except ValueError:
                # Handle other value errors (this is a bit more general)
                print("Value error encountered while parsing JSON.")
                topk_dicts = {'top_1': '', 'top_2': '', 'top_3': '', 'top_4': '', 'top_5': ''}
            except Exception as e:
                # Catch-all for any other unexpected errors
                print(f"Unexpected error while parsing JSON: {e}")
                topk_dicts = {'top_1': '', 'top_2': '', 'top_3': '', 'top_4': '', 'top_5': ''} 

            print("topk_dicts: ", topk_dicts)
            return dict_to_list(topk_dicts)

        filtered_resps = [apply_to_resp([resp]) for resp in results][0][0]

        return {"topk_acc": topk_match_fn(gold, filtered_resps) }


    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {k: mean for k in ["topk_acc"]}

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {k: True for k in ["topk_acc"]}
