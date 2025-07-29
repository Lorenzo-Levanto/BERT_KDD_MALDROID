import torch.nn as nn
import torch
# Custom Model ispirato a sentinelbert gandalf
class BertMultiClassificationHeads(nn.Module):
    def __init__(self, bert_model, layer_sizes, dropout_probs):
        super(BertMultiClassificationHeads, self).__init__()
        self.bert_model = bert_model
        assert len(layer_sizes) - 1 == len(dropout_probs), "Mismatch between layer sizes and dropout probabilities"
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_probs[i]))

        #self.output_layer = nn.Linear(layer_sizes[-1], 1)
        self.output_layer = nn.Linear(layer_sizes[-1], 5)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs[1] 
        for layer in self.layers:
            x = layer(x)
        output_layer = self.output_layer(x)
        #output_layer = torch.sigmoid(output_layer)
        return output_layer#.squeeze(1)#squeeze1 non serve?     
    
 