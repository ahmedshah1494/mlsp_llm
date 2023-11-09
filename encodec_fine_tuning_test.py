import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio, concatenate_datasets
from torchaudio.transforms import MelSpectrogram
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
from multiprocessing import cpu_count
class AutoRegressiveAudioEncoderConfig(PretrainedConfig):
    def __init__(self, encodec_model_name="facebook/encodec_24khz",
                 lm_model_name='facebook/opt-125m', recon_loss_weight=1.0,
                 ppl_loss_weight=1.0, num_quantizers=2, encodec_kwargs={}, llm_kwargs={}, **kwargs):
        super().__init__(**kwargs)
        self.encodec_model_name = encodec_model_name
        self.lm_model_name = lm_model_name
        self.recon_loss_weight = recon_loss_weight
        self.ppl_loss_weight = ppl_loss_weight
        self.num_quantizers = num_quantizers
        self.encodec_kwargs = encodec_kwargs
        self.llm_kwargs = llm_kwargs

class AutoRegressiveAudioEncoder(PreTrainedModel):
    config_class = AutoRegressiveAudioEncoderConfig
    is_parallelizable = False
    supports_gradient_checkpointing = True
    
    def __init__(self, config: config_class) -> None:
        super().__init__(config)
        self.config = config
        self.audio_encoder = EncodecModel.from_pretrained(config.encodec_model_name,
                                                          differentiable_quantization=True,
                                                          low_cpu_mem_usage=False,
                                                          **config.encodec_kwargs)
        self.processor = AutoProcessor.from_pretrained(config.encodec_model_name, low_cpu_mem_usage=False)

        self.llm = OPTForCausalLM(OPTConfig.from_pretrained(config.lm_model_name, vocab_size=self.audio_encoder.config.codebook_size, max_position_embeddings=3072, low_cpu_mem_usage=False, **config.llm_kwargs))
    
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncodecModel, OPTForCausalLM)):
            module.gradient_checkpointing = value
    
    def to(self, device):
        super().to(device)
        self.audio_encoder.to(device)
        self.llm.to(device)
        return self
    
    def forward(self, input_values, padding_mask, *args, **kwargs):
        encoder_outputs = self.audio_encoder.encode(input_values, padding_mask)
        audio_codes = encoder_outputs.audio_codes
        audio_codes = audio_codes[:, :, :self.config.num_quantizers]
        
        if (self.config.recon_loss_weight > 0.0) or (not self.training):
            recon_loss = 0.
            audio_values = self.audio_encoder.decode(audio_codes, encoder_outputs.audio_scales, padding_mask)[0]
            if self.audio_encoder.config.compute_discriminator_loss or self.audio_encoder.config.train_discriminator:
                logits_ref, fmap_ref = self.audio_encoder.disc(input_values)
                logits_pred, fmap_pred = self.audio_encoder.disc(audio_values)
                disc_outputs = (logits_ref, logits_pred, fmap_ref, fmap_pred)
                if self.audio_encoder.config.train_discriminator:
                    logit_pred_detached = self.audio_encoder.disc(audio_values.detach())[0]
                    discriminator_loss = self.audio_encoder._compute_discriminator_loss(logits_ref, logit_pred_detached)
                    recon_loss += discriminator_loss
            else:
                audio_values = None
                disc_outputs = None
            
            if self.audio_encoder.config.train_encoder or self.audio_encoder.config.train_decoder:
                reconstruction_loss = self.audio_encoder._compute_reconstruction_loss(input_values, audio_values,
                                                                                      encoder_outputs.commitment_loss,
                                                                                      disc_outputs)
                recon_loss += reconstruction_loss

            # recon_loss_wav = self._get_wav_loss(encoder_outputs.audio_values, input_values)
            # recon_loss_mel = self._get_freq_loss(encoder_outputs.audio_values, input_values)
            # recon_loss = 0.1*recon_loss_wav + recon_loss_mel + encoder_outputs.commitment_loss
            # print(recon_loss_wav.item(), recon_loss_mel.item(), encoder_outputs.commitment_loss.item())
        else:
            recon_loss = torch.tensor(0.0).to(input_values.device)

        if (self.config.ppl_loss_weight > 0.0) or (not self.training):
            llm_embedding_matrix = self.llm.get_input_embeddings().weight
            audio_codes = audio_codes.squeeze(0).reshape(-1, audio_codes.shape[-2], audio_codes.shape[-1])
            audio_embed_llm = torch.matmul(audio_codes, llm_embedding_matrix)
            labels = audio_codes.argmax(-1).detach().clone()
            # print(audio_embed_llm.shape, labels.shape, labels.max(), labels.min())
            ppl = self.llm(inputs_embeds=audio_embed_llm, labels=labels).loss
        else:
            ppl = torch.tensor(0.0).to(input_values.device)
        loss = self.config.recon_loss_weight * recon_loss + self.config.ppl_loss_weight * ppl

        if np.random.rand() < (1/100):
            print(f"recon_loss: {recon_loss.item()}, ppl: {ppl.item()}")
        if not torch.isfinite(loss):
            print(f"recon_loss: {recon_loss.item()}, ppl: {ppl.item()}")
            raise ValueError("Loss is not finite")
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
    parser.add_argument('--disc_ckp', type=str, default='encodec_disc_ckpt.pt')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--lm_model_name', type=str, default='facebook/opt-125m')
    parser.add_argument('--recon_loss_weight', type=float, default=1.0)
    parser.add_argument('--ppl_loss_weight', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='librispeech_asr')
    parser.add_argument('--dataset_split', type=str, nargs='+', default="train.clean.100")
    parser.add_argument('--dataset_subset', type=str, default=None)
    parser.add_argument('--lm_pretraining', action='store_true')
    parser.add_argument('--disc_pretraining', action='store_true')
    parser.add_argument('--update_disc', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_quantizers', type=int, default=2)
    parser.add_argument('--push_to_hub', action='store_true')
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--ignore_size_mismatch_in_ckp', action='store_true')
    args = parser.parse_args()
    config = AutoRegressiveAudioEncoderConfig(lm_model_name=args.lm_model_name,
                                              num_quantizers=args.num_quantizers,
                                              recon_loss_weight=args.recon_loss_weight,
                                              ppl_loss_weight=args.ppl_loss_weight)
    if args.ckp_name is not None:
        print(f'Loading model from {args.ckp_name}')
        model = AutoRegressiveAudioEncoder.from_pretrained(f"{args.ckp_name}",
                                              num_quantizers=args.num_quantizers,
                                              recon_loss_weight=args.recon_loss_weight,
                                              ppl_loss_weight=args.ppl_loss_weight,
                                              ignore_mismatched_sizes=args.ignore_size_mismatch_in_ckp)
    else:
        model = AutoRegressiveAudioEncoder(config)

    if (args.disc_ckp is not None) and (args.disc_ckp != ''):
        print(f'Loading discriminator from {args.disc_ckp}')
        model.audio_encoder.disc.load_state_dict(torch.load(args.disc_ckp)['model_state_dict'])

    if args.lm_pretraining:
        model.config.recon_loss_weight = 0.0
        model.audio_encoder.requires_grad_(False)
    elif args.disc_pretraining:
        model.audio_encoder.config.train_discriminator = True
        model.audio_encoder.config.train_encoder = False
        model.audio_encoder.config.train_decoder = False
        model.config.ppl_loss_weight = 0.0
        model.requires_grad_(False)
        model.audio_encoder.disc.requires_grad_(True)
    else:
        model.audio_encoder.config.train_encoder = True
        model.audio_encoder.config.train_decoder = True
        model.config.recon_loss_weight = args.recon_loss_weight
        model.requires_grad_(True)
        model.audio_encoder.disc.requires_grad_(args.update_disc)
    print(f'lm_pretraining: {args.lm_pretraining}, recon_loss_weight: {model.config.recon_loss_weight}, ppl_loss_weight: {model.config.ppl_loss_weight}')
    model = model.to('cuda:0')
    dataset = [load_dataset(args.dataset, args.dataset_subset, split=split) for split in args.dataset_split]
    if len(dataset) == 1:
        dataset = dataset[0]
    else:
        dataset = concatenate_datasets(dataset)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=model.processor.sampling_rate))
    print(dataset, len(dataset))

    def collate_fn(batch):
        raw_audio = [elem['audio']['array'] for elem in batch]
        inputs = model.processor(raw_audio=raw_audio, sampling_rate=model.processor.sampling_rate, return_tensors="pt")
        return inputs

    suffix = []
    if args.lm_pretraining:
        suffix.append('lm_pretraining')
    if args.disc_pretraining:
        suffix.append('disc_pretraining')
    if args.ckp_name is not None:
        suffix.append('pretrained-ft')
    else:
        suffix.append('e2e')
    suffix = '-'.join(suffix)
    model_name=f"{config.encodec_model_name.split('/')[-1]}-{config.lm_model_name.split('/')[-1]}-{suffix}{args.exp_name}"
    print(f"Model name: {model_name}")
    train_args = TrainingArguments(
        output_dir=f"/ocean/projects/cis220031p/mshah1/mlsp_llm/saves/autoreg_encodec_ft/{model_name}",
        do_train=True,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=model.supports_gradient_checkpointing,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        save_total_limit=1,
        remove_unused_columns=False,
        logging_steps=100,
        log_level="info",
        push_to_hub=args.push_to_hub,
        hub_model_id=f"cmu-mlsp/{model_name}",
        hub_strategy='end',
        save_strategy='steps',
        save_steps=50,
        dataloader_num_workers=cpu_count(),
        dataloader_drop_last=True,
    )
    trainer = Trainer(model, train_args, train_dataset=dataset, eval_dataset=dataset, data_collator=collate_fn)
    trainer.train(resume_from_checkpoint=args.resume_training)
