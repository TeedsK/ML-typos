# quick_test_model.py (optional)
from edit_tag_spellfix.model import RobertaTagger
import torch, json

tag2id = json.load(open("data/60k_roberta/tag2id.json"))
model = RobertaTagger.from_pretrained_with_tags(
    base_model_name="roberta-base",
    tag2id_path="data/60k_roberta/tag2id.json",
)
dummy = torch.randint(0, model.config.vocab_size, (2, 16))
out = model(dummy, attention_mask=torch.ones_like(dummy))
print(out["logits"].shape)   # -> torch.Size([2,16,num_tags])
