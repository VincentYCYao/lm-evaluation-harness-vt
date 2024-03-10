import json
import collections
import re
import string

from lm_eval.api.task import ConfigurableTask
from lm_eval.api.metrics import mean

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):  # inputs are strings
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def topk_match_fn(gold, preds):
    preds, gold = [pred.lower() for pred in preds], [target.lower() for target in gold]
    hits = 0
    for pred in preds:
        if pred in gold:
            hits += 1
            break
    return hits


def compute_f1_from_lists(gold, preds):
    # f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)

    f1_scores = [0] * len(preds)
    for idx, pred in enumerate(preds):
        f1_scores[idx] += max(compute_f1(a, pred) for a in gold)

    return max(f1_scores)


class BioLAMA_umls(ConfigurableTask):
    DATASET_PATH = "JadeCheng/Biolama-umls"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0  # "Yaml"
    DATASET_NAME = None

    # CONFIG = None
    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})

    def doc_to_text(self, doc):
        # umls
        pid2prompt_meta = {'UR44': {'template': '[X] treats [Y] .'},
                           'UR221': {'template': '[X] has a genetic association with [Y] .'},
                           'UR45': {'template': '[X] treats [Y] .'}, 'UR48': {'template': '[X] results in [Y] .'},
                           'UR211': {'template': '[X] involves [Y] .'}, 'UR214': {'template': '[Y] causes [X] .'},
                           'UR256': {'template': '[Y] has a genetic association with [X] .'},
                           'UR588': {'template': '[X] involves [Y] process .'},
                           'UR254': {'template': '[X] has symptoms such as [Y] .'},
                           'UR180': {'template': '[Y] is finding of disease [X] .'},
                           'UR116': {'template': '[X] is clinically associated with [Y] .'},
                           'UR625': {'template': '[X] has a genetic association with [Y] .'},
                           'UR46': {'template': '[X] should not be used in the presence of [Y] disease .'},
                           'UR173': {'template': '[X] is caused by [Y] .'},
                           'UR49': {'template': '[X] has a mechanism of action of [Y] .'},
                           'UR50': {'template': '[X] is a therapeutic class of [Y] .'},
                           'UR124': {'template': 'The most widely used drug for preventing [X] is [Y] .'}}
        template = pid2prompt_meta[doc["predicate_id"]]["template"]
        subject = doc["sub_label"]
        sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

        prefix = f'Consider the following sentence: "{sentence}"'
        suffix = ('\n\n-> Which noun-phrase should <BLANK> be filled with? '
                  'Give me 5 most probable candidates. '
                  'Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", '
                  'where the value for key "top_1" is the most promising entity that would replace <BLANK>.')

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

        gold = self.doc_to_target(doc)  # lower_objects

        def dict_to_list(topk_dict, k=5):

            ordered_list = []

            # Loop through keys from top_1 to top_k
            for i in range(1, k + 1):
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

        return {"topk_acc": topk_match_fn(gold, filtered_resps), "f1": compute_f1_from_lists(gold, filtered_resps)}

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


class BioLAMA_umls_test(ConfigurableTask):
    DATASET_PATH = "CamWheeler135/Bio_Lama_TinyUMLS"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0  # "Yaml"
    DATASET_NAME = None

    # CONFIG = None
    def __init__(self):
        super().__init__(config={'metadata': {'version': self.VERSION}})

    def doc_to_text(self, doc):
        # umls
        pid2prompt_meta = {'UR44': {'template': '[X] treats [Y] .'},
                           'UR221': {'template': '[X] has a genetic association with [Y] .'},
                           'UR45': {'template': '[X] treats [Y] .'}, 'UR48': {'template': '[X] results in [Y] .'},
                           'UR211': {'template': '[X] involves [Y] .'}, 'UR214': {'template': '[Y] causes [X] .'},
                           'UR256': {'template': '[Y] has a genetic association with [X] .'},
                           'UR588': {'template': '[X] involves [Y] process .'},
                           'UR254': {'template': '[X] has symptoms such as [Y] .'},
                           'UR180': {'template': '[Y] is finding of disease [X] .'},
                           'UR116': {'template': '[X] is clinically associated with [Y] .'},
                           'UR625': {'template': '[X] has a genetic association with [Y] .'},
                           'UR46': {'template': '[X] should not be used in the presence of [Y] disease .'},
                           'UR173': {'template': '[X] is caused by [Y] .'},
                           'UR49': {'template': '[X] has a mechanism of action of [Y] .'},
                           'UR50': {'template': '[X] is a therapeutic class of [Y] .'},
                           'UR124': {'template': 'The most widely used drug for preventing [X] is [Y] .'}}
        template = pid2prompt_meta[doc["predicate_id"]]["template"]
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

        gold = self.doc_to_target(doc)  # lower_objects

        def dict_to_list(topk_dict, k=5):

            ordered_list = []

            # Loop through keys from top_1 to top_k
            for i in range(1, k + 1):
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

        return {"topk_acc": topk_match_fn(gold, filtered_resps)}

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


class BioLAMA_Wikidata(ConfigurableTask):
    DATASET_PATH = "JadeCheng/Biolama-wikidata"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0  # "Yaml"
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
        template = pid2prompt_meta[doc["predicate_id"]]["template"]

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

        gold = self.doc_to_target(doc)  # lower_objects

        def dict_to_list(topk_dict, k=5):

            ordered_list = []

            # Loop through keys from top_1 to top_k
            for i in range(1, k + 1):
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

        return {"topk_acc": topk_match_fn(gold, filtered_resps)}

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


class BioLAMA_Wikidata_test(ConfigurableTask):
    DATASET_PATH = "CamWheeler135/Bio_Lama_TinyWikidata"
    OUTPUT_TYPE = 'generate_until'
    VERSION = 0.0  # "Yaml"
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
        template = pid2prompt_meta[doc["predicate_id"]]["template"]

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

        gold = self.doc_to_target(doc)  # lower_objects

        def dict_to_list(topk_dict, k=5):

            ordered_list = []

            # Loop through keys from top_1 to top_k
            for i in range(1, k + 1):
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

        return {"topk_acc": topk_match_fn(gold, filtered_resps)}

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

class BioLAMA_ctd(ConfigurableTask):
        DATASET_PATH = "JadeCheng/Biolama-ctd"
        OUTPUT_TYPE = 'generate_until'
        VERSION = 0.0  # "Yaml"
        DATASET_NAME = None

        # CONFIG = None

        def __init__(self):
            super().__init__(config={'metadata': {'version': self.VERSION}})

        def doc_to_text(self, doc):
            # manual prompt template
            pid2prompt_meta = {
                'CD1': {'template': '[X] prevents diseases such as [Y].'},
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

            # Make the template for that specific doc.
            template = pid2prompt_meta[doc["predicate_id"]]["template"]

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

            gold = self.doc_to_target(doc)  # lower_objects

            def dict_to_list(topk_dict, k=5):

                ordered_list = []

                # Loop through keys from top_1 to top_k
                for i in range(1, k + 1):
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

            return {"topk_acc": topk_match_fn(gold, filtered_resps)}

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

class BioLAMA_ctd_test_prompt(ConfigurableTask):
            DATASET_PATH = "JadeCheng/Biolama-ctd"
            OUTPUT_TYPE = 'generate_until'
            VERSION = 0.0  # "Yaml"
            DATASET_NAME = None

            # CONFIG = None

            def __init__(self):
                super().__init__(config={'metadata': {'version': self.VERSION}})

            def doc_to_text(self, doc):
                # manual prompt template
                pid2prompt_meta = {
                    'CD1': {'template': '[X] prevents diseases such as [Y].'},
                    'CD2': {
                        'template': '[X] exposure is associated with significant increases in diseases such as [Y].'},
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

                # Make the template for that specific doc.
                template = pid2prompt_meta[doc["predicate_id"]]["template"]

                subject = doc["sub_label"]
                sentence = template.replace('[X]', subject).replace('[Y]', "<BLANK>")

                prefix = f'Consider the following sentence: "{sentence}"'

                # baseline prompt
                suffix = '\n\n-> Which noun-phrase should <BLANK> be filled with? Give me 5 most probable candidates. Output your response in JSON format with keys "top_1", "top_2", "top_3", "top_4" and "top_5", where the value for key "top_1" is the most promising entity that would replace <BLANK>.'
                # try new prompt

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

                gold = self.doc_to_target(doc)  # lower_objects

                def dict_to_list(topk_dict, k=5):

                    ordered_list = []

                    # Loop through keys from top_1 to top_k
                    for i in range(1, k + 1):
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

                return {"topk_acc": topk_match_fn(gold, filtered_resps)}

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
