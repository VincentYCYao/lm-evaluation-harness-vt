some changes:

__main__.py: add the argument --save_results, if True, save, or else, no, replace this file with the old __main__.py
evaluator.py: pass the argument --save_results, and implement the save_result option, also replace this 

medlama: 

1. implement the medlama task, notice: for the medlama, they didn't split the dataset into train, dev and test, similar to the Biolama (roughly calculate), I manually split it as 4:1:5, and changed the orginal csv file into jsonl to keep identical with Biolama

2. add the instance variables in the class file, "self.unfiltered", "self.prediction" and "self.hit" for saving, also replace this and fit with your task

evaluator.py: for the save function, add the unfiltered response, but save only for the output_type: generate_untill


example command line:
lm_eval --model openai-chat-completions --model_args model=gpt-3.5-turbo --tasks /raid/home/s2521923/Pycharm/lm-evaluation-harness/lm_eval/tasks/medlama/ --save_results True --output_path ./eval_output
