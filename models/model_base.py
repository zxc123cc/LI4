from torch import nn
from modeling_roberta import RobertaForSequenceClassification
from modeling_roberta import RobertaForMaskedLM
from transformers import RobertaTokenizer
import torch

class DialFactClassification(nn.Module):

    def __init__(self, model_dir):
        super().__init__()
        self.base_model = RobertaForSequenceClassification.from_pretrained(model_dir)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        outputs = self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)
        return outputs


class PromptModel(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.model = RobertaForMaskedLM.from_pretrained(model_dir)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                *args,
                **kwargs):
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        # outputs = outputs['logits'][torch.where(labels>0)]
        # outputs = outputs.view(labels.shape[0], -1, outputs.shape[1])
        # if outputs.shape[1] == 1:
        #     outputs = outputs.view(outputs.shape[0], outputs.shape[2])

        return outputs

class PromptForClassification(nn.Module):
    def __init__(self,model_dir,verbalizer):
        super().__init__()
        self.prompt_model = PromptModel(model_dir)
        self.verbalizer = verbalizer
    def extract_at_mask(self,
                        outputs: torch.Tensor,
                        loss_ids):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(loss_ids>0)]
        outputs = outputs.view(loss_ids.shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None):
        outputs = self.prompt_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict)

        outputs = self.verbalizer.gather_outputs(outputs)
        if isinstance(outputs, tuple):
            outputs_at_mask = [self.extract_at_mask(output, labels) for output in outputs]
        else:
            outputs_at_mask = self.extract_at_mask(outputs, labels)
        label_words_logits = self.verbalizer.process_outputs_zxc(outputs_at_mask)
        return label_words_logits