import torch.nn as nn    

class BERT_architecture(nn.Module):

    def __init__(self, bert):
        super(BERT_architecture, self).__init__()
        self.bert = bert
        
        self.dropout = nn.Dropout(0.3)  # Increased dropout for regularization
        self.layer_norm = nn.LayerNorm(768)  # Layer normalization
        
        self.fc1 = nn.Linear(768, 256)  # Dense layer
        self.fc2 = nn.Linear(256, 3)  # Output layer with 3 classes
        
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask, token_type_ids):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        x = self.layer_norm(cls_hs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x