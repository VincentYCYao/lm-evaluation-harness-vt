some changes:

__main__.py: add the argument --save_results, if True, save, or else, no, replace this file with the old __main__.py
evaluator.py: pass the argument --save_results, and implement the save_result option, also replace this 

biolama_ctd: add the instance variables in the class file, "self.prediction" and "self.hit" for saving, also replace this and fit with your task

Notice!!!!

I just implemented the function for the output_type: generate_until. Later I'll do it for the output_types we need like multi-choices.
And for the current save, I just save the filtered_resp, which means if your model response is not a valid json formate, the csv file will save the None dict for that question (later I'll save the unfiltered one)

example command line:
lm_eval --model openai-chat-completions --model_args model=gpt-3.5-turbo --tasks /raid/home/s2521923/Pycharm/lm-evaluation-harness/lm_eval/tasks/biolama_ctd/ --save_results True --output_path ./eval_output
