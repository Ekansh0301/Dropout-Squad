# narrative_critic.py
import torch
import torch.nn as nn
from transformers import DebertaV2PreTrainedModel, DebertaV2Model
from transformers.modeling_outputs import SequenceClassifierOutput

class NarrativeCritic(DebertaV2PreTrainedModel):
    """Narrative Quality Critic for evaluating descriptive richness and coherence"""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1
        self.config = config
        
        self.deberta = DebertaV2Model(config)
        self.regressor = nn.Linear(config.hidden_size, self.num_labels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Filter out unexpected arguments
        deberta_kwargs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        outputs = self.deberta(**deberta_kwargs)
        
        # Use [CLS] token for regression
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        
        # Raw outputs for MSE loss (sigmoid applied later)
        loss = None
        if labels is not None:
            # Ensure labels are float for MSE loss
            labels = labels.float()
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.view(-1), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,  # Raw outputs, not sigmoid
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def predict_quality(self, text, tokenizer, device):
        """Convenience method for quality prediction"""
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self(**inputs)
            
            # Apply sigmoid here for final prediction to get [0,1] range
            predicted_score = torch.sigmoid(outputs.logits).item()
            return predicted_score