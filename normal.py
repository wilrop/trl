import tempfile
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

train_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")
eval_dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="test")

def dummy_reward_func(completions, **kwargs):
    return [0.0] * len(completions)

with tempfile.TemporaryDirectory() as tmp_dir:
    training_args = GRPOConfig(
        output_dir=tmp_dir,
        learning_rate=0.1,  # increase the learning rate to speed up the test
        per_device_train_batch_size=3,  # reduce the batch size to reduce memory usage
        num_generations=3,  # reduce the number of generations to reduce memory usage
        max_completion_length=32,  # reduce the completion length to reduce memory usage
        report_to="none",
        max_steps=5,
        dispatch_batches=False,
    )
    trainer = GRPOTrainer(
        model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        reward_funcs=dummy_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
