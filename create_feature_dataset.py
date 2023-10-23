import torch
import transformers
transformers.logging.set_verbosity_info()
from encodec_fine_tuning_test import AutoRegressiveAudioEncoder, AutoRegressiveAudioEncoderConfig
from datasets import load_dataset, Audio
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckp_name', type=str, required=False)
    parser.add_argument('--dataset', type=str, default='librispeech_asr')
    parser.add_argument('--dataset_subset', type=str, default=None)
    parser.add_argument('--dataset_split', type=str, default="train.clean.100")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--push_to_hub', action='store_true')
    args = parser.parse_args()

    if args.ckp_name is not None:
        model = AutoRegressiveAudioEncoder.from_pretrained(f"{args.ckp_name}")
    else:
        model = AutoRegressiveAudioEncoder(AutoRegressiveAudioEncoderConfig())
        args.ckp_name = 'encodec_24khz'
    if os.path.exists(args.ckp_name):
        model_name = args.ckp_name.split('/')[2]
    else:
        model_name = args.ckp_name.split('/')[-1]
    dataset_name = f"{model_name}-{args.dataset.split('/')[-1]}-{args.dataset_split}-features"
    print(f'dataset name: {dataset_name}')
    
    processor = model.processor
    model = model.audio_encoder.to('cuda:0')

    dataset = load_dataset(args.dataset, args.dataset_subset, split=args.dataset_split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))

    @torch.no_grad()
    def get_features(batch):
        raw_audio = [audio['array'] for audio in batch['audio']]
        inputs = processor(raw_audio=raw_audio, sampling_rate=processor.sampling_rate, return_tensors="pt").to('cuda:0')
        codes = model.encode(**inputs).audio_codes.squeeze(0).argmax(-1)
        batch['audio_codes'] = codes.detach().cpu().numpy()
        return batch
    
    dataset = dataset.map(get_features, batched=True, batch_size=args.batch_size)

    
    dataset.save_to_disk(f"/ocean/projects/cis220031p/mshah1/mlsp_llm/saves/features/{dataset_name}")
    dataset.push_to_hub(f"cmu-mlsp/{dataset_name}")