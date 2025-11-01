import os
import json
import wandb
from transformers import TrainerCallback
from transformers import Trainer, TrainingArguments
from math import ceil
import shutil

from model.utils import save_results_json, log_results_to_wandb, evaluate_model_checkpoint, save_best_model_creativity
from model.eval import save_train_nll, visualize_attention_weights


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            print(f"[Step {state.global_step}] Loss: {logs.get('loss')}")

class LiveEvalCallback(TrainerCallback):
    def __init__(self, eval_fn, log_fn, every_n_steps, temperatures, num_eval_runs, decode_fn, train_dataset, results_path, seed_len, model_save_metric="creativity_mean", eval_batch_size=16):
        self.eval_fn = eval_fn
        self.log_fn = log_fn
        self.every_n_steps = every_n_steps
        self.temperatures = temperatures
        self.num_eval_runs = num_eval_runs
        self.decode_fn = decode_fn
        self.train_dataset = train_dataset
        self.results_path = results_path
        self.seed_len = seed_len
        self.eval_batch_size = eval_batch_size  

        self.results_json = os.path.join(self.results_path, f"HL{seed_len}.json") if self.results_path is not None else None
        self.best_model_path = os.path.join(self.results_path, "best_models", f"HL{seed_len}") if self.results_path is not None else None
        self.nll_hist_path = os.path.join(self.results_path, f"nll_hist_HL{seed_len}") if self.results_path is not None else None
        self.attention_heatmap_path = os.path.join(self.results_path, f"attention_heatmaps_HL{seed_len}") if self.results_path is not None else None
        
        if self.results_path is not None: 
            # if os.path.exists(self.results_path):
            #     # Delete the entire directory tree
            #     shutil.rmtree(self.results_path)
            #     print(f"Removed existing results directory and all its contents: {self.results_path}")

            # # Create the fresh, empty directory
            # os.makedirs(self.results_path)
            # print(f"Created empty directory: {self.results_path}")
            os.makedirs(self.results_path, exist_ok=True)

            os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)
            os.makedirs(self.nll_hist_path, exist_ok=True)
            os.makedirs(self.attention_heatmap_path, exist_ok=True)

        if not results_path is None and os.path.exists(self.results_json):
            with open(self.results_json, "r") as f:
                self.all_results = json.load(f)
        else:
            self.all_results = {}
        
        self.model_save_metric = model_save_metric  # Metric to track for best model
        self.best_score = float('-inf')

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step > 0 and step % self.every_n_steps == 0:
            if self.eval_fn is None or self.results_path is None:
                # If no eval function or results path is provided, skip evaluation
                print("No evaluation function or results path provided. Skipping live evaluation.")
                return
            print(f"\n📈 Live evaluation at step {step}")
            results, step_best_creativity_temp, step_best_creativity_score, corresponding_mem = evaluate_model_checkpoint(
                model, step, self.temperatures, self.num_eval_runs, self.decode_fn, self.train_dataset, self.eval_fn
            )
            if self.log_fn is not None:
                self.log_fn(step, results)
            self.all_results[str(step)] = results

            save_results_json(self.all_results, self.results_json)

            save_train_nll(model, dataset=self.train_dataset, tokenizer=self.train_dataset.tokenizer, num_samples=1000, 
                           step=step, nll_hist_path=self.nll_hist_path)
            
            self.best_score = save_best_model_creativity(model=model, tokenizer=self.train_dataset.tokenizer, step_best_score=step_best_creativity_score, 
                                                         best_score=self.best_score, save_path=self.best_model_path)
            
            visualize_attention_weights(model, tokenizer=self.train_dataset.tokenizer, train_dataset=self.train_dataset, 
                                        temperature=step_best_creativity_temp, num_samples=1000, max_length=self.train_dataset[1]["labels"].shape[0]*2, 
                                        step=step, save_path=self.attention_heatmap_path, batch_size=self.eval_batch_size, 
                                        creativity=step_best_creativity_score, memorization=corresponding_mem)
            print(f"✅ Evaluation logged and saved at step {step}")
            
        model.train()  # Ensure model is in train mode after evaluation

def train_model(model, dataset, data_collator, device, batch_size, num_epochs, eval_callback, num_workers, lr, gradient_accumulation_steps):
    model = model.to(device)
    training_args = TrainingArguments(
        output_dir=None,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        logging_steps=20,
        dataloader_num_workers=num_workers,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        learning_rate=lr,
        gradient_accumulation_steps=gradient_accumulation_steps
        # weight_decay=0.01,           # A standard regularization value
    
        # # --- Cosine Annealing Scheduler ---
        # lr_scheduler_type='cosine',  # This is the key change from StepLR
        # warmup_ratio=0.1,            # Use 10% of steps for warmup
        
        # # --- Gradient Clipping ---
        # max_grad_norm=1.0, 
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=[eval_callback],
    )
    trainer.train()
    return model

def train_main(
    model,
    dataset_name,
    save_name,
    batch_size,
    num_epochs,
    temperatures,
    num_eval_runs,
    train_dataset,
    data_collator,
    device,
    decode_fn,
    eval_fn,
    num_checkpoints=100,
    save_results=True,
    num_workers=32,
    log_to_wandb=True,
    lr=5e-5,
    eval_batch_size=32,
    gradient_accumulation_steps=1
):
    seed_len = train_dataset.seed_len
    if save_results:
        results_path = f"/datastor1/vansh/lang_sampling/results/{dataset_name}/{save_name}"
        last_model_path = os.path.join(results_path, "last_models", f"HL{seed_len}") if results_path is not None else None

        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        if log_to_wandb:
            wandb.init(project=f"{dataset_name}-{save_name}", name="in-memory-eval", config={
                "temperatures": temperatures,
                "eval_metric1": "representation_power",
                "eval_metric2": "creativity",
                "HL": seed_len,
            })
            log_fn = log_results_to_wandb
        else:
            log_fn = None
    else:
        results_path = None
        log_fn = None
        
    total_steps = ceil(len(train_dataset) / batch_size) * num_epochs
    every_n_steps = max(1, total_steps // num_checkpoints)

    eval_callback = LiveEvalCallback(
        eval_fn=eval_fn,
        log_fn=log_fn,
        every_n_steps=every_n_steps,
        temperatures=temperatures,
        num_eval_runs=num_eval_runs,
        decode_fn=decode_fn,
        train_dataset=train_dataset,
        results_path=results_path,
        seed_len=seed_len,
        eval_batch_size=eval_batch_size
    )

    train_model(
        model,
        dataset=train_dataset,
        data_collator=data_collator,
        device=device,
        batch_size=batch_size,
        num_epochs=num_epochs,
        eval_callback=eval_callback,
        num_workers=num_workers,
        lr=lr,
        gradient_accumulation_steps=gradient_accumulation_steps
    )

    if last_model_path:
        print(f"Saving the last model to {last_model_path}")
        model.save_pretrained(last_model_path, safe_serialization=False)
        if train_dataset.tokenizer:
            train_dataset.tokenizer.save_pretrained(last_model_path)

    print("\n✅ Live evaluation complete. Results saved and logged to wandb.")


# class LiveEvalCallback_per_epoch(TrainerCallback):
#     def __init__(self, eval_fn, log_fn, every_n_epochs, temperatures, num_eval_runs, decode_fn, train_dataset, results_path, seed_len):
#         self.eval_fn = eval_fn
#         self.log_fn = log_fn
#         self.every_n_epochs = every_n_epochs
#         self.temperatures = temperatures
#         self.num_eval_runs = num_eval_runs
#         self.decode_fn = decode_fn
#         self.train_dataset = train_dataset
#         self.results_path = results_path
#         self.seed_len = seed_len
#         self.results_json = os.path.join(results_path, f"HL{seed_len}.json")
#         if os.path.exists(self.results_json):
#             with open(self.results_json, "r") as f:
#                 self.all_results = json.load(f)
#         else:
#             self.all_results = {}

#     def on_epoch_end(self, args, state, control, model=None, **kwargs):
#         epoch = int(state.epoch)
#         if epoch % self.every_n_epochs == 0:
#             print(f"\n📈 Live evaluation at epoch {epoch}")
#             model.eval()  # Ensure model is in eval mode
#             with torch.no_grad():
#                 results = evaluate_model_checkpoint(
#                     model, epoch, self.temperatures, self.num_eval_runs, self.decode_fn, self.train_dataset, self.eval_fn
#                 )

#                 self.log_fn(epoch, results)
#                 self.all_results[str(epoch)] = results
#                 save_results_json(self.all_results, self.results_json)
#                 print(f"✅ Evaluation logged and saved at epoch {epoch}")
#             # Switch back to train mode
#             model.train()  
