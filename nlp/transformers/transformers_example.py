import torch

import transformers
from transformers import AdamW
from transformers import get_constant_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer

class NLPDataset:
    def __init__(self, data, tokenizer, max_len):
        self.excerpt = data["text"]
        self.target = data["target"]
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.excerpt)

    def __getitem__(self, item):
        text = str(self.excerpt[item])
        target = self.target[item]
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
    
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }


class AttentionBlock(nn.Module):
    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features

        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)

    def forward(self, features):
        att = torch.tanh(self.W(features))

        score = self.V(att)

        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class NLPModel(transformers.PreTrainedModel):
    def __init__(self, conf, model_name):
        super(NLPModel, self).__init__(conf)
        self.model = AutoModel.from_pretrained(model_name, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.att = AttentionBlock(768, 768, 1)
        self.l0 = nn.Linear(768 * 1, 1)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask):
        outputs = self.model(
            ids,
            attention_mask=mask,
        )
        out = outputs['hidden_states']
        out = out[-1]
        out = self.drop_out(out)
        out = torch.mean(out, 1, True)  # ここmeanでいいのか？
        #out = self.att(out)
        
        out = self.l0(out)
        out = out.squeeze(-1).squeeze(-1)
        
        return out

class NLPModelForSequenceClassification(transformers.PreTrainedModel):
    def __init__(self, conf, model_name):
        super(NLPModelForSequenceClassification, self).__init__(conf)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        # self.att = AttentionBlock(self.in_features, self.in_features, 1)
        # self.fc = nn.Linear(self.in_features, 1)

    def forward(self, ids, mask):
        outputs = self.model(
            ids,
            attention_mask=mask,
        )
        out = outputs['logits'].squeeze(-1)
        return out

class NLPModelForTokenClassification(transformers.PreTrainedModel):
    def __init__(self, conf, model_name):
        super(NLPModelForTokenClassification, self).__init__(conf)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=1)
        # self.att = AttentionBlock(self.in_features, self.in_features, 1)
        # self.fc = nn.Linear(self.in_features, 1)

    def forward(self, ids, mask):
        outputs = self.model(
            ids,
            attention_mask=mask,
        )
        out = outputs['logits']
        out = torch.mean(out, 1, True)
        out = out.squeeze(-1).squeeze(-1)
        # out = self.att(out)
        # out = self.fc(out)
        return out
