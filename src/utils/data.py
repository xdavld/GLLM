import json
import random
from typing import List
from datasets import Dataset, load_dataset

from pprint import pprint

def get_data(args):
    """
    Load the dataset based on the provided arguments.

    Args:
        args: The arguments containing dataset name and other configurations.

    Returns:
        The loaded dataset.
    """
    random.seed(args["seed"])

    datasets = load_dataset(
        "csv",
        data_files={"train": args["dataset_path"]},
    )

    datasets["train"] = datasets["train"].shuffle(seed=args["seed"])
    
    datasets["train"] = apply_prompt_template(operation=args["operation"], dataset=datasets["train"], prompt_template_path=args["prompt_template_path"], sample_size=args["sample_size"], output_size=args["output_size"])

    if args["operation"] == "fine-tuning":
        datasets = split_dataset(datasets, dataset_splits=args["dataset_splits"], eval_size=args["eval_size"], test_size=args["test_size"])

    return datasets


def apply_prompt_template(operation: str, dataset: Dataset, prompt_template_path: str, sample_size: int, output_size: int):
    """
    Apply the prompt template to the dataset.

    Args:
        dataset: The dataset to which the prompt template will be applied.
        prompt_template: The prompt template to apply.

    Returns:
        The dataset with the applied prompt template.
    """

    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read()

    json_strings = [
        json.dumps(example, ensure_ascii=False) for example in dataset
    ]

    if operation == "synthesization":
        chunks = [
            json_strings[i : i + sample_size]
            for i in range(0, len(json_strings) - sample_size + 1, sample_size)
        ]

        prompts = []
        input_jsons = []
        for chunk in chunks:
            input_json = "[" + ",".join(chunk) + "]"
            input_jsons.append(input_json)
            prompt = (
                template
                .replace("{{input}}", input_json)
                .replace("{{output_size}}", str(output_size))
            )
            prompts.append(prompt)
        
        return Dataset.from_dict({"prompt": prompts, "data": input_jsons})
    
    elif operation == "training":
        chunks = [
            json_strings[i : i + sample_size]
            for i in range(0, len(json_strings) - sample_size + 1, sample_size)
        ]

        prompts = []
        input_jsons = []
        for chunk in chunks:
            sample = random.choice(chunk)
            input_json_sample = f"[{sample}]"
            input_jsons.append(input_json_sample)

            input_json = "[" + ",".join(chunk) + "]"
            prompt = (
                template
                .replace("{{input}}", input_json)
                .replace("{{output_size}}", str(output_size))
            )
            prompts.append(prompt)
        
        return Dataset.from_dict({"prompt": prompts, "data": input_jsons})
    
    elif operation == "fine-tuning":
        chunks = [
            json_strings[i : i+sample_size+output_size]
            for i in range(0, len(json_strings) - (sample_size+output_size) + 1, sample_size+output_size)
        ]

        prompts = []
        completions = []
        for chunk in chunks:
            prompt_examples = chunk[:sample_size]
            completion_examples = chunk[sample_size:]

            prompt_json = "[" + ",".join(prompt_examples) + "]"
            completion_json = "[" + ",".join(completion_examples) + "]"

            prompt_text = (
                template
                .replace("{{input}}", prompt_json)
                .replace("{{output_size}}", str(output_size))
            )

            prompts.append(prompt_text)
            completions.append(completion_json)
        
        return Dataset.from_dict({"prompt": prompts, "completion": completions})


def split_dataset(dataset: Dataset, dataset_splits: List[str] = ["train", "test"], eval_size: float = 0.2, test_size: float = 0.1):
    """
    Split the loaded DatasetDict into train, eval, and test sets.

    Args:
        dataset: A DatasetDict with at least the "train" split.
        dataset_splits: 
            - ["train"] 
            - ["train", "eval"] 
            - ["train", "eval", "test"]
        eval_size: Fraction for the eval split (only when creating eval or test).
        test_size: Fraction for the test split (only when creating eval+test).

    Returns:
        A dict with keys equal to dataset_splits, each mapping to a Dataset.
    """

    if dataset_splits == ["train"]:
        return {"train": dataset["train"]}

    elif dataset_splits == ["train", "eval"]:
        split = dataset["train"].train_test_split(
            test_size=eval_size
        )
        return {"train": split["train"], "eval": split["test"]}

    elif dataset_splits == ["train", "eval", "test"]:
        train_test = dataset["train"].train_test_split(
            test_size=eval_size + test_size
        )
        eval_test = split_datasets["test"].train_test_split(
            test_size=test_size / (eval_size + test_size)
        )
        return {
            "train": train_test["train"],
            "eval": eval_test["train"],
            "test": eval_test["test"]
        }

    else:
        raise ValueError(f"Invalid dataset splits: {dataset_splits}. Supported splits are ['train'], ['train', 'eval'], or ['train', 'eval', 'test'].")