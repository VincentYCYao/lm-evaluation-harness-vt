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


class MedLAMA(ConfigurableTask):
    DATASET_PATH = "JadeCheng/MedLAMA"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0 # "Yaml"
    DATASET_NAME = None
    # CONFIG = None

    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})
        self.unfiltered = None
        self.prediction = None
        self.hit = 0

    def doc_to_text(self, doc):

        pid2prompt_meta = {
            'associated_morphology_of': {'template': '[X] is associated morphology of [Y] .'},
            'disease_has_abnormal_cell': {'template': '[X] has the abnormal cell [Y] .'},
            'disease_has_associated_anatomic_site': {'template': 'The disease [X] can stem from the associated anatomic_site [Y] .'},
            'disease_has_normal_cell_origin': {'template': 'The disease [X] stems from the normal cell [Y] .'},
            'disease_has_normal_tissue_origin': {'template': 'The disease [X] stems from the normal tissue [Y] .'},
            'disease_mapped_to_gene': {'template': 'The disease [X] is mapped to gene [Y] .'},
            'disease_may_have_associated_disease': {'template': 'The disease [X] might have the associated disease [Y] .'},
            'disease_may_have_finding': {'template': '[X] may have [Y] .'},
            'disease_may_have_molecular_abnormality': {'template': 'The disease [X] may have molecular abnormality [Y] .'},
            'gene_associated_with_disease': {'template': 'The gene [X] is associatied with disease [Y] .'},
            'gene_encodes_gene_product': {'template': 'The gene [X] encodes gene product [Y] .'},
            'gene_product_encoded_by_gene': {'template': 'The gene product [X] is encoded by gene [Y] .'},
            'gene_product_has_associated_anatomy': {'template': 'The gene product [X] has the associated anatomy [Y] .'},
            'gene_product_has_biochemical_function': {'template': '[X] has biochemical function [Y] .'},
            'gene_product_has_chemical_classification': {'template': 'The gene product [X] is a type of [Y] .'},
            'gene_product_plays_role_in_biological_process': {'template': 'The gene product [X] plays role in biological process [Y] .'},
            'has_physiologic_effect': {'template': '[X] has physiologic effect of [Y] .'},
            'may_prevent': {'template': '[X] may be able to prevent [Y] .'},
            'may_treat': {'template': '[X] might treat [Y] .'},
            'occurs_after': {'template': '[X] occurs after [Y] .'}}

        # GPT
        template =  pid2prompt_meta[doc["rel"]]["template"]
        head = doc["head_name"]
        sentence = template.replace('[X]', head).replace('[Y]', "<BLANK>")

        prefix = f'Consider the following sentence: "{sentence}"'
        suffix = '\n\n-> Which noun-phrase should <BLANK> be filled with? Give me 5 most probable candidates. Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", where the value for key "top_1" is the most promising entity that would replace <BLANK>.'

        prompt = prefix + suffix
        return f"{prompt}\n"


    def doc_to_target(self, doc):
        # print(doc['tail_names_list'])
        tails = eval(doc['tail_names_list'])
        # tails = json.loads(doc['tail_names_list'].replace("'", '"'))
        # sample = json.loads(line)
        # print("tails", type(tails))
        lower_tails = list(dict.fromkeys([tail.lower() for tail in tails]))
        # print(lower_tails)
        # print(doc['tail_cuis_len'])
        assert len(lower_tails) == int(doc['tail_cuis_len'])
        print(f"lower_tails = {lower_tails}")

        return lower_tails

        # # print(f"lower_objects = {lower_objects}")
        #
        # return lower_objects

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
        self.unfiltered = results
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

