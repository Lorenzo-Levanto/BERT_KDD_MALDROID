import torch.nn as nn

# Custom Model ispirato a sentinelbert gandalf
class BertBinaryClassificationHeads(nn.Module):
    def __init__(self, bert_model, layer_sizes, dropout_probs):
        super(BertBinaryClassificationHeads, self).__init__()
        self.bert_model = bert_model
        assert len(layer_sizes) - 1 == len(dropout_probs), "Mismatch between layer sizes and dropout probabilities"
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_probs[i]))

        self.output_layer = nn.Linear(layer_sizes[-1], 2)  # 2 output per classificazione binaria

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs[1] 
        for layer in self.layers:
            x = layer(x)
        logits = self.output_layer(x)
        return logits  #restituisce logits per cross!!!
