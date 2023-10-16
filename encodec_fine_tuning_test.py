import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
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

class AutoRegressiveAudioEncoderConfig(PretrainedConfig):
    def __init__(self, recon_loss_weight=1.0, ppl_loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.recon_loss_weight = recon_loss_weight
        self.ppl_loss_weight = ppl_loss_weight

class AutoRegressiveAudioEncoder(PreTrainedModel):
    config_class = AutoRegressiveAudioEncoderConfig

    def __init__(self, config: config_class) -> None:
        super().__init__(config)
        self.config = config
        self.audio_encoder = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

        self.llm = OPTForCausalLM(OPTConfig.from_pretrained("facebook/opt-1.3b"))
    
    def forward(self, inputs, labels=None):
        # inputs = self.processor(raw_audio=inputs, sampling_rate=self.processor.sampling_rate, return_tensors="pt").to(self.device)
        encoder_outputs = self.audio_encoder.encode(inputs["input_values"], inputs["padding_mask"])
        audio_values = self.audio_encoder.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]

        recon_loss = torch.nn.functional.mse_loss(audio_values, inputs["input_values"])
        ppl = self.llm(encoder_outputs.audio_codes, labels=labels).loss
        loss = self.config.recon_loss_weight * recon_loss + self.config.ppl_loss_weight * ppl
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'ppl': ppl,
            'encoder_outputs': encoder_outputs,
            'audio_values': audio_values,
        }

config = AutoRegressiveAudioEncoderConfig()
model = AutoRegressiveAudioEncoder(config)
dataset = load_dataset("librispeech_asr", "clean", split="train.100")
dataset = dataset.cast_column("audio", Audio(sampling_rate=model.processor.sampling_rate))

def transform(batch):
    raw_audio = [audio['array'] for audio in batch['audio']]
    new_batch = model.processor(raw_audio, sampling_rate=model.processor.sampling_rate, return_tensors="pt")
    return new_batch

dataset = dataset.map(transform, batched=True, batch_size=128, num_proc=4, load_from_cache_file=True)
print(dataset[0], len(dataset))
train_args = TrainingArguments(
    output_dir="/ocean/projects/cis220031p/mshah1/mlsp_llm/saves/autoreg_encodec_ft/",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    warmup_ratio=0.1,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    save_total_limit=1,
)
trainer = Trainer(model, train_args, train_dataset=dataset, data_collator=DataCollatorWithPadding(model.processor, padding=True))
trainer.train()