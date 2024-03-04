import json
import os
from typing import Union, List

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_task
from lm_eval.api.metrics import mean
# from lm_eval.api.registry import register_metric


# eval_logger = logging.getLogger("lm-eval")

# @register_metric(
#     metric="acc_at_topk_ctd",
#     higher_is_better=True,
#     # output_type=["loglikelihood", "multiple_choice"],
#     output_type="generate_until",
#     aggregation="mean",
# )
def topk_match_fn(gold, preds):

    hits = 0
    for pred in preds:
        if pred in gold:
            hits += 1 
    # print("hits", hits)
    return hits


class BioLAMA(ConfigurableTask):
    DATASET_PATH = "JadeCheng/Biolama-ctd"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0 # "Yaml"
    DATASET_NAME = None
    # CONFIG = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})
        self.prediction = None
        self.hit = 0

    def doc_to_text(self, doc):
        pid2prompt_meta = {
            'CD1': {'template': '[X] prevents diseases such as [Y].'},
            # 'CD1': {'template': 'What kinds of diseases that {subject_entity} can prevent?'},
            'CD2': {'template': '[X] exposure is associated with significant increases in diseases such as [Y].'},
            'CG1': {'template': '[X] treatment decreases the levels of [Y] expression.'},
            'CG17': {'template': '[X] treatment increases the levels of [Y] expression.'},
            'CG18': {'template': '[X] upregulates [Y] protein.'},
            'CG2': {'template': '[X] results in decreased activity of [Y] protein.'},
            'CG21': {'template': '[X] results in increased phosphorylation of [Y] protein.'},
            'CG4': {'template': '[X] results in increased activity of [Y] protein.'},
            'CG6': {'template': '[X] treatment decreases the levels of [Y] expression.'},
            'CG9': {'template': '[X] binds to [Y] protein.'},
            'CP1': {'template': '[X] analog results in decreased phenotypes such as [Y] .'},
            'CP2': {'template': '[X] induces phenotypes such as [Y].'},
            'CP3': {'template': '[X] affects phenotypes such as [Y].'},
            'GD1': {'template': 'Gene [X] is associated with diseases such as [Y] .'},
            'GP1': {'template': 'Gene [X] is associated with pathways such as [Y].'}
        }
        # umls
        # pid2prompt_meta = {'UR44': {'template': '[X] treats [Y] .'}, 'UR221': {'template': '[X] has a genetic association with [Y] .'}, 'UR45': {'template': '[X] treats [Y] .'}, 'UR48': {'template': '[X] results in [Y] .'}, 'UR211': {'template': '[X] involves [Y] .'}, 'UR214': {'template': '[Y] causes [X] .'}, 'UR256': {'template': '[Y] has a genetic association with [X] .'}, 'UR588': {'template': '[X] involves [Y] process .'}, 'UR254': {'template': '[X] has symptoms such as [Y] .'}, 'UR180': {'template': '[Y] is finding of disease [X] .'}, 'UR116': {'template': '[X] is clinically associated with [Y] .'}, 'UR625': {'template': '[X] has a genetic association with [Y] .'}, 'UR46': {'template': '[X] should not be used in the presence of [Y] disease .'}, 'UR173': {'template': '[X] is caused by [Y] .'}, 'UR49': {'template': '[X] has a mechanism of action of [Y] .'}, 'UR50': {'template': '[X] is a therapeutic class of [Y] .'}, 'UR124': {'template': 'The most widely used drug for preventing [X] is [Y] .'}}
        # wikidata
        # pid2prompt_meta = {'P2176': {'template': 'The standard treatment for patients with [X] is a drug such as [Y].'}, 'P2175': {'template': '[X] has effects on diseases such as [Y].'}, 'P4044': {'template': '[X] cures diseases such as [Y].'}, 'P780': {'template': '[X] has symptoms such as [Y].'}, 'P2293': {'template': 'Gene [X] has a genetic association with diseases such as [Y].'}}
        # system_prompt = f"You are a helpful, respectful and honest assistant. You need to answer the given question less than 3 words.\n\n Example 1: \n Question: What kinds of diseases that bacterial toxin can prevent? \n Answer: Colonic Neoplasms. \n Your answer should be less than 3 words. Do not give any explainations."

        # GPT
        template =  pid2prompt_meta[doc["predicate_id"]]["template"]
        subject = doc["sub_label"]
        sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

        prefix = f'Consider the following sentence: "{sentence}"'
        suffix = '\n\n-> Which noun-phrase should <BLANK> be filled with? Give me 5 most probable candidates. Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", where the value for key "top_1" is the most promising entity that would replace <BLANK>.'

        prompt = prefix + suffix
        return f"{prompt}\n"

        # for llama
        # base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"

        # template = pid2prompt_meta[doc["predicate_id"]]["template"]
        # subject = doc["sub_label"]
        # sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")
        #
        # prompt = "'<s>[INST]\n<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>> {{}} [/INST]'"
        #
        # prefix = f'Consider the following sentence: "{sentence}"'
        # suffix = '\n\n-> Which noun-phrase should <BLANK> be filled with? Give me 5 most probable candidates. Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", where the value for key "top_1" is the most promising entity that would replace <BLANK>.'
        # input = prefix + suffix
        # prompt = f"'<s>[INST]\n<<SYS>> You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  <</SYS>> {{input}} [/INST]'"

        # return f"{prompt}\n"


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
        # result_dict = {}
        # use_metric = list(self._metric_fn_list.keys())
        # metric = 'acc@k'
        gold = self.doc_to_target(doc) # lower_objects
        # question = self.doc_to_text(doc)
        print("result", results[0])
        def dict_to_list(topk_dict, k=5):

            ordered_list = []

            # Loop through keys from top_1 to top_k
            for i in range(1, k+1):
                key = "top_" + str(i)
                if key in topk_dict:
                    ordered_list.append(topk_dict[key])
                        
            return ordered_list


        def apply_to_resp(resp): 

            # print("resp_app", resp)
            # print("json", json.loads(resp[0]))
            # print("topk_dict", [json.loads(resp[0])])
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
                # return None
            # except ValueError:
            #     # print("The first string is not valid JSON.")
            #     topk_dicts = {'top_1': '', 'top_2': '', 'top_3': '', 'top_4': '', 'top_5': ''}
            # topk_dicts = [topk_dicts]
            # topk_dicts = [json.loads(json_str) for json_str in resp]
            print("topk_dicts: ", topk_dicts)
            return dict_to_list(topk_dicts)
            # return [dict_to_list(topk_dict) for topk_dict in topk_dicts]

        # filtered_resps = apply_to_resp([results[0]])
        filtered_resps = [apply_to_resp([resp]) for resp in results][0]#[0]
        # for metric in self._metric_fn_list.keys():
        # result_dict["topk_acc"] =  0
        # result_score = acc_topk_match_fn(gold, filtered_resps)
        # # result_socre =
        # result_dict["topk_acc"] = result_score
        # return result_dict
        self.prediction = filtered_resps
        result_score = topk_match_fn(gold, filtered_resps)
        self.hit = result_score
        return {"topk_acc": result_score} #, gold, question, result_score


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

