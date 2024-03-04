import argparse
import os
import sys
import json

sys.path.append(os.getcwd())

from kubejobs.jobs import KubernetesJob


def argument_parser():
    parser = argparse.ArgumentParser(description="Harness Job Runner")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--namespace", type=str, default="eidf106ns")
    parser.add_argument("--base_image", type=str, default="ghcr.io/camwheeler135/lm-harness:v0.1.1")
    parser.add_argument("--cpu", type=str, default="10")
    parser.add_argument("--ram", type=str, default="64Gi")
    parser.add_argument("--n_gpu", type=int, default=2)
    parser.add_argument("--gpu_product", type=str, default="NVIDIA-A100-SXM4-40GB")
    parser.add_argument("--queue_name", type=str, default="eidf106ns-user-queue")
    parser.add_argument("--config_file", type=str, help="JSON file that contains the configuration names.")
    parser.add_argument("--pvc", type=str, default="vt-pvc")
    args = parser.parse_args()
    return args


def main():

    args = argument_parser()
    with open(args.config_file, 'r') as config_file:
        configs = json.load(config_file)

    #
    cmd1 = "cd /app && "\
	   "mkdir /app/lm-eval-log && "\
	   "apt -y update && apt -y upgrade && "\
           "apt -y install git python3 && "\
	   "pip install -U pip && "\
	   "git clone git@github.com:VincentYCYao/lm-evaluation-harness-vt.git && "\
	   "cd /app/lm-evaluation-harness-vt && "\
	   "pip install -e ."

    cmd2 = "python lm_eval --model hf "\
	   f"--model_args pretrained={configs['model']},parallelize=True "\
	   f"--tasks {configs['task']} "\
	   f"--device cuda "\
	   f"--batch_size auto "\
	   f"--output_path /app/lm-eval-log/{configs['save_file_name']}.jsonl "\
	   "--log_samples"

    # Create a Kubernetes Job with a name, container image, and command
    print(f"Creating job for: {args.run_name}")
    job = KubernetesJob(
        name=args.run_name,
        cpu_request=args.cpu,
        ram_request=args.ram,
        image=args.base_image,
        gpu_type="nvidia.com/gpu",
        gpu_limit=args.n_gpu,
        gpu_product=args.gpu_product,
        backoff_limit=0,
	command=["/bin/bash", "-c", "--"],
	args=[cmd1 + cmd2],
        #command=["/opt/conda/bin/conda", "run", "-n", "lm_env", "bash", "experiment_runners/run_experiment-vt.sh", f"{configs['model']}", f"{configs['task']}", f"{configs['save_file_name']}", f"{configs['input_path']}"],
        namespace=args.namespace,
        kueue_queue_name=args.queue_name,
        volume_mounts={"data-mount": {"mountPath": "/app/lm-eval-log/", "pvc": args.pvc} }
    )
    
    job.generate_yaml()
    print(job.generate_yaml())

    # Run the Job on the Kubernetes cluster
    job.run()


if __name__ == "__main__":
    main()
