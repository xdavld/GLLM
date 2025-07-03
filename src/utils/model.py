import re
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMGAN:
    def __init__(self, generator: str, discriminator: str, dis_prompt: str):
        self.generator = Generator(model=generator)
        self.discriminator = Discriminator(model=discriminator)
        self.dis_prompt = dis_prompt
    
    def __call__(self, data, **kwargs):
        return self.generate(data, **kwargs)
    
    def format_discriminator_data(self, real_data, generated_texts):
        labeled_pairs = []

        for sample in real_data:
            prompt = self.dis_prompt.replace("{{sample}}", sample)
            labeled_pairs.append((prompt, 1))

        for sample in generated_texts:
            prompt = self.dis_prompt.replace("{{sample}}", sample)
            labeled_pairs.append((prompt, 0))

        random.shuffle(labeled_pairs)

        prompts, labels = zip(*labeled_pairs)
        return list(prompts), list(labels)
    
    def train(self, data, gen_generation_params, **kwargs):
        self.generator.set_mode("train")
        self.generator.set_alpha(kwargs.get("alpha", 0.5))
        self.discriminator.set_mode("train")

        self.generator.set_optimizer(
            optimizer=torch.optim.AdamW(
                self.generator.get_model().parameters(), 
                lr=float(kwargs.get("generator_lr", 2e-05))
            )
        )
        self.discriminator.set_optimizer(
            optimizer=torch.optim.AdamW(
                self.discriminator.get_model().parameters(),
                lr=float(kwargs.get("discriminator_lr", 2e-05))
            )
        )

        num_epochs = kwargs.get("num_epochs", 1)
        batch_size = kwargs.get("batch_size", 1)
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]

                generated_texts = self.generator.generate(batch["prompt"], gen_generation_params)
                prompts, labels = self.format_discriminator_data(real_data=batch["data"], generated_texts=generated_texts)
                results = self.discriminator.predict(prompts, labels)
                gen_loss = self.generator.train(generated_texts, results["rewards"])

            output_dir = kwargs.get("output_dir", "../output")

            output_dir_gen = output_dir+"/Generator/chekpoint-"+str(epoch+1)
            self.generator.get_model().save_pretrained(output_dir_gen)
            self.generator.get_tokenizer().save_pretrained(output_dir_gen)

            output_dir_dis = output_dir+"/Discriminator/chekpoint-"+str(epoch+1)
            self.discriminator.get_model().save_pretrained(output_dir_dis)
            self.discriminator.get_tokenizer().save_pretrained(output_dir_dis)

            print("{"+f"Epoch: {epoch+1}, Generator Loss: {gen_loss}, Discriminator Loss: {results['loss']}, Discriminator Accuracy: {results['accuracy']}"+"}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
        def generate(self, data, gen_generation_params, **kwargs):
            self.generator.set_mode("generate")
            self.discriminator.set_mode("generate")
    
            generated_texts = self.generator.generate(data["prompt"], gen_generation_params)
            prompts, labels = self.format_discriminator_data(real_data=data["data"], generated_texts=generated_texts)
            results = self.discriminator.predict(prompts, labels)
        
            print("{"+f"Discriminator Accuracy: {results['accuracy']}"+"}")
    
            output_dir = kwargs.get("output_dir", "../output")
            
            # Save the generated texts as .txt
            with open(f"{output_dir}/generated_texts.txt", "w") as f:
                for text in generated_texts:
                    f.write(text + "\n")
    
            # Parse JSON strings and save as CSV
            rows = []
            for text in generated_texts:
                try:
                    # Each text is a JSON object or list of objects
                    data_list = json.loads(text)
                    if isinstance(data_list, dict):
                        data_list = [data_list]
                    rows.extend(data_list)
                except Exception as e:
                    print(f"Error parsing JSON: {e}\nText: {text}")
    
            # Save all generated data
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(f"{output_dir}/generated_data_all.csv", index=False)
            else:
                print("No valid data to save as CSV found.")
    
            # Save only samples predicted as real by the discriminator
            if rows and "labels" in results:
                pred_labels = results["labels"]
                real_indices = [i for i, label in enumerate(pred_labels) if label == 1]
                real_rows = [rows[i] for i in real_indices if i < len(rows)]
                if real_rows:
                    df_real = pd.DataFrame(real_rows)
                    df_real.to_csv(f"{output_dir}/generated_data_real.csv", index=False)
                else:
                    print("No samples predicted as real by the discriminator.")
        

class GANModel(ABC):
    def __init__(self, model: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.mode = None
        self.optimizer = None
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def set_mode(self, mode):
        if mode not in ["generate", "train"]:
            raise ValueError("Mode must be either 'generate' or 'train'.")
        self.mode = mode

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
    
    @abstractmethod
    def train(self):
        pass


class Generator(GANModel):
    def __init__(self, model: str):
        super().__init__(model)
        self.alpha = None
    
    def __call__(self, inputs, generation_params = None):
        return self.generate(inputs, generation_params)
    
    def set_alpha(self, alpha):
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be between 0 and 1.")
        self.alpha = alpha

    def generate(self, prompts, generation_params = None):
        self.model.eval()

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **(generation_params or {}))
        
        generated_texts = []
        for output in outputs:
            new_tokens = output[inputs['input_ids'].shape[-1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            generated_texts.append(text)
        
        return generated_texts
    
    def json_str_value(self, tok):
        return (
            re.match(r'^".*"$', tok) or
            re.match(r'^\d+(\.\d+)?$', tok) or
            re.match(r'^\.\d+$', tok) or
            tok in ['true', 'false', 'null']
        )
    
    def compute_grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

    def train(self, generated_texts, rewards):
        if self.mode != "train":
            raise ValueError("Model must be in 'train' mode to train.")

        self.model.train()

        masked_ce_labels = []
        masked_rl_positions = []
        input_ids_list = []

        for text in generated_texts:
            enc = self.tokenizer(text, return_tensors="pt")
            input_ids = enc.input_ids[0]
            labels = input_ids.clone()
            rl_mask = torch.zeros_like(labels, dtype=torch.bool)  # marks value tokens

            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            for i, tok in enumerate(tokens):
                if self.json_str_value(tok):
                    labels[i] = -100  # don't use in CE
                    rl_mask[i] = True  # but use in RL

            input_ids_list.append(input_ids)
            masked_ce_labels.append(labels)
            masked_rl_positions.append(rl_mask)

        # Pad
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(masked_ce_labels, batch_first=True, padding_value=-100)
        rl_mask_padded = pad_sequence(masked_rl_positions, batch_first=True, padding_value=False)

        input_ids_padded = input_ids_padded.to(self.model.device)
        labels_padded = labels_padded.to(self.model.device)
        rl_mask_padded = rl_mask_padded.to(self.model.device)

        # Forward pass
        outputs = self.model(input_ids=input_ids_padded, labels=labels_padded)
        ce_loss = outputs.loss

        logits = outputs.logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        rl_loss_total = 0.0
        count = 0

        for b in range(len(generated_texts)):
            target_ids = input_ids_padded[b]
            rl_mask = rl_mask_padded[b]
            log_probs_b = log_probs[b]

            # Only consider RL tokens
            rl_indices = torch.where(rl_mask)[0]
            if len(rl_indices) == 0:
                continue

            selected_log_probs = log_probs_b[rl_indices, target_ids[rl_indices]]
            reward = rewards[b]

            rl_loss_total += (-reward * selected_log_probs.mean())
            count += 1

        rl_loss = rl_loss_total / max(1, count)  # avoid division by zero

        # Compute CE grad norm
        self.model.zero_grad()
        ce_loss.backward(retain_graph=True)
        ce_grad_norm = self.compute_grad_norm(self.model)
        self.model.zero_grad()

        # Compute RL grad norm
        rl_loss.backward(retain_graph=True)
        rl_grad_norm = self.compute_grad_norm(self.model)
        self.model.zero_grad()

        # Scale RL loss
        beta = ce_grad_norm / (rl_grad_norm + 1e-8)

        # Final combined loss
        loss = (1 - self.alpha) * ce_loss + self.alpha * beta * rl_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()


class Discriminator(GANModel):
    def __init__(self, model: str):
        super().__init__(model)
        self.token_zero_id = self.tokenizer.convert_tokens_to_ids("0")
        self.token_one_id = self.tokenizer.convert_tokens_to_ids("1")
    
    def __call__(self, inputs, **kwargs):
        return self.predict(inputs, **kwargs)
    
    def predict(self, prompts, labels):
        self.model.train() if self.mode == "train" else self.model.eval()

        targets = [str(label) for label in labels]

        # Tokenize prompts and targets separately
        prompt_encodings = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        target_encodings = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False
        )

        # Concatenate input and target
        input_ids_list = []
        labels_list = []

        for i in range(len(prompts)):
            prompt_ids = prompt_encodings["input_ids"][i]
            target_ids = target_encodings["input_ids"][i]

            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            label_ids = torch.full_like(input_ids, -100) 
            label_ids[-len(target_ids):] = target_ids

            input_ids_list.append(input_ids)
            labels_list.append(label_ids)

        # Pad to same length
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        input_ids_padded = input_ids_padded.to(self.model.device)
        labels_padded = labels_padded.to(self.model.device)

        with torch.set_grad_enabled(self.mode == "train"):
            outputs = self.model(input_ids=input_ids_padded, labels=labels_padded)

            if self.mode == "train":
                loss = outputs.loss
                self.train(loss=loss)
            else:
                loss = None

            # Get last token logits for prediction
            last_logits = outputs.logits[:, -1, :]

            # Extract and normalize logits
            logits_two = torch.stack([
                last_logits[:, self.token_zero_id],
                last_logits[:, self.token_one_id]
            ], dim=-1)
            probs = F.softmax(logits_two, dim=-1)

            pred_labels = torch.argmax(probs, dim=-1)
            rewards = (probs[:, 1] - probs[:, 0]).tolist()

        return {
            "labels": pred_labels.cpu().tolist(),
            "accuracy": (pred_labels == torch.tensor(labels, device=pred_labels.device)
).float().mean().item() if labels is not None else None,
            "rewards": rewards,
            "loss": loss.item() if loss is not None else None
        }

    
    def train(self, loss):
        if self.mode != "train":
            raise ValueError("Model must be in 'train' mode to train.")

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()