import torch
import spacy
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from torchtext.vocab import build_vocab_from_iterator

SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

def get_spacy_tokenizers():
    import subprocess
    def load_or_download(lang_model, cmd):
        try:
            return spacy.load(lang_model)
        except OSError:
            subprocess.run(cmd, check=True)
            return spacy.load(lang_model)
            
    german_nlp = load_or_download("de_core_news_sm", ["python", "-m", "spacy", "download", "de_core_news_sm"])
    english_nlp = load_or_download("en_core_web_sm", ["python", "-m", "spacy", "download", "en_core_web_sm"])
    return german_nlp, english_nlp

def tokenise_german(sentence, nlp_model):
    return [token.text.lower() for token in nlp_model(sentence)]

def tokenise_english(sentence, nlp_model):
    return [token.text.lower() for token in nlp_model(sentence)]

class Multi30kDataset(Dataset):
    def __init__(self, split="train", src_vocab=None, tgt_vocab=None):
        self.split = split
        self.de_nlp, self.en_nlp = get_spacy_tokenizers()
        
        dataset_hf = load_dataset("bentrevett/multi30k", trust_remote_code=True)
        self.raw_data = dataset_hf[split]
        
        if split == "train":
            if src_vocab is not None or tgt_vocab is not None:
                pass
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            assert src_vocab is not None and tgt_vocab is not None, "Vocabs required for non-train splits."
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
            
        self.samples = self.process_data()

    def build_vocab(self):
        def yield_de():
            for item in self.raw_data:
                yield tokenise_german(item["de"], self.de_nlp)
                
        def yield_en():
            for item in self.raw_data:
                yield tokenise_english(item["en"], self.en_nlp)

        v_src = build_vocab_from_iterator(yield_de(), specials=SPECIALS, special_first=True)
        v_src.set_default_index(UNK_IDX)

        v_tgt = build_vocab_from_iterator(yield_en(), specials=SPECIALS, special_first=True)
        v_tgt.set_default_index(UNK_IDX)

        return v_src, v_tgt

    def process_data(self):
        processed = []
        for row in self.raw_data:
            de_tokens = tokenise_german(row["de"], self.de_nlp)
            en_tokens = tokenise_english(row["en"], self.en_nlp)

            de_indices = [SOS_IDX] + self.src_vocab(de_tokens) + [EOS_IDX]
            en_indices = [SOS_IDX] + self.tgt_vocab(en_tokens) + [EOS_IDX]

            processed.append((torch.tensor(de_indices, dtype=torch.long), torch.tensor(en_indices, dtype=torch.long)))
        return processed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

def collate_fn(batch, pad_idx=PAD_IDX):
    src_list, tgt_list = zip(*batch)
    padded_src = pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
    padded_tgt = pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
    return padded_src, padded_tgt

def build_dataloaders(batch_size=128):
    train_ds = Multi30kDataset(split="train")
    src_v = train_ds.src_vocab
    tgt_v = train_ds.tgt_vocab

    val_ds = Multi30kDataset(split="validation", src_vocab=src_v, tgt_vocab=tgt_v)
    test_ds = Multi30kDataset(split="test", src_vocab=src_v, tgt_vocab=tgt_v)

    my_collate = lambda b: collate_fn(b, pad_idx=PAD_IDX)

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=my_collate)
    vl_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=my_collate)
    ts_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=my_collate)

    return tr_loader, vl_loader, ts_loader, src_v, tgt_v