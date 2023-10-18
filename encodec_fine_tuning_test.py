import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
import transformers
transformers.logging.set_verbosity_info()
from transformers import (
    EncodecModel,
    AutoProcessor,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from tqdm import tqdm
import numpy as np
import argparse

class AutoRegressiveAudioEncoderConfig(PretrainedConfig):
    def __init__(self, encodec_model_name="facebook/encodec_24khz", lm_model_name='facebook/opt-125m', recon_loss_weight=1.0, ppl_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encodec_model_name = encodec_model_name
        self.lm_model_name = lm_model_name
        self.recon_loss_weight = recon_loss_weight
        self.ppl_loss_weight = ppl_loss_weight

class AutoRegressiveAudioEncoder(PreTrainedModel):
    config_class = AutoRegressiveAudioEncoderConfig

    def __init__(self, config: config_class) -> None:
        super().__init__(config)
        self.config = config
        self.supports_gradient_checkpointing = True
        self.audio_encoder = EncodecModel.from_pretrained(config.encodec_model_name, differentiable_quantization=True)
        self.processor = AutoProcessor.from_pretrained(config.encodec_model_name)

        self.llm = OPTForCausalLM(OPTConfig.from_pretrained(config.lm_model_name, vocab_size=self.audio_encoder.config.codebook_size, max_position_embeddings=3072))
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncodecModel, OPTForCausalLM)):
            module.gradient_checkpointing = value
    
    def to(self, device):
        super().to(device)
        self.audio_encoder.to(device)
        self.llm.to(device)
        print(self.audio_encoder.device, self.llm.device, self.device)
        return self
    
    def forward(self, raw_audio, *args, **kwargs):
        inputs = self.processor(raw_audio=raw_audio, sampling_rate=self.processor.sampling_rate, return_tensors="pt").to(self.device)
        encoder_outputs = self.audio_encoder.encode(inputs["input_values"], inputs["padding_mask"])
        audio_codes = encoder_outputs.audio_codes
        audio_values = self.audio_encoder.decode(audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
        recon_loss = torch.nn.functional.mse_loss(audio_values, inputs["input_values"])
        
        llm_embedding_matrix = self.llm.get_input_embeddings().weight
        audio_codes = audio_codes.squeeze(0).reshape(-1, audio_codes.shape[-2], audio_codes.shape[-1])
        audio_embed_llm = torch.matmul(audio_codes, llm_embedding_matrix)
        labels = audio_codes.argmax(-1).detach().clone()
        # print(audio_embed_llm.shape, labels.shape, labels.max(), labels.min())
        ppl = self.llm(inputs_embeds=audio_embed_llm, labels=labels).loss
        loss = self.config.recon_loss_weight * recon_loss + self.config.ppl_loss_weight * ppl

        if np.random.rand() < (1/100):
            print(f"recon_loss: {recon_loss.item()}, ppl: {ppl.item()}")
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'ppl': ppl,
            'encoder_outputs': encoder_outputs,
            'audio_values': audio_values,
        }
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--lm_model_name', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='librispeech_asr')
    parser.add_argument('--dataset_split', type=str, default="train.clean.100")
    parser.add_argument('--dataset_subset', type=str, default=None)
    parser.add_argument('--lm_pretraining', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--push_to_hub', action='store_true')
    args = parser.parse_args()
    config = AutoRegressiveAudioEncoderConfig(lm_model_name=args.lm_model_name)
    if args.ckp_name is not None:
        model = AutoRegressiveAudioEncoder.from_pretrained(f"{args.ckp_name}")
    else:
        model = AutoRegressiveAudioEncoder(config)

    if args.lm_pretraining:
        model.config.recon_loss_weight = 0.0
        model.audio_encoder.requires_grad_(False)
    else:
        model.requires_grad_(True)
    model = model.to('cuda:0')
    dataset = load_dataset(args.dataset, args.dataset_subset, split=args.dataset_split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=model.processor.sampling_rate))
    print(dataset, len(dataset))

    def collate_fn(batch):
        new_batch = {
            'raw_audio': [elem['audio']['array'] for elem in batch],
        }
        return new_batch

    if args.lm_pretraining:
        suffix = 'lm_pretraining'
    elif args.ckp_name is not None:
        suffix = 'pretrained-ft'
    else:
        suffix = 'e2e'
    model_name=f"{config.encodec_model_name.split('/')[-1]}-{config.lm_model_name.split('/')[-1]}-{suffix}{args.exp_name}"
    train_args = TrainingArguments(
        output_dir=f"/ocean/projects/cis220031p/mshah1/mlsp_llm/saves/autoreg_encodec_ft/{model_name}",
        do_train=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        save_total_limit=1,
        remove_unused_columns=False,
        logging_steps=100,
        log_level="info",
        push_to_hub=args.push_to_hub,
        hub_model_id=f"cmu-mlsp/{model_name}",
        hub_strategy='end',
        save_strategy='epoch',
        fsdp=False,
    )
    trainer = Trainer(model, train_args, train_dataset=dataset, eval_dataset=dataset, data_collator=collate_fn)
    trainer.train()
