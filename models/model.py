from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers.file_utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import SequenceClassifierOutput,MaskedLMOutput
from transformers.models.roberta.modeling_roberta import  ROBERTA_INPUTS_DOCSTRING, \
    _TOKENIZER_FOR_DOC, _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC
import torch
from transformers.utils import logging
from transformers.activations import gelu

from component import IGM
import numpy

logger = logging.get_logger(__name__)

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()

        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()


        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)

        weights = torch.matmul(queries, keys.transpose(1, 2))
        weights = weights / (self.input_dim ** 0.5)
        weights = torch.softmax(weights, dim=-1)

        outputs = torch.matmul(weights, values)

        return outputs.squeeze(1)


class DialFactClassification(nn.Module):

    def __init__(self, model_dir,num_labels=3,is_IGM=True,is_LabelEmb=True, iter_num=3, LabelEmb_dim=128):
        super().__init__()
        self.base_model = RobertaForSequenceClassification.from_pretrained(
            model_dir,num_labels,is_IGM,is_LabelEmb,iter_num,LabelEmb_dim
        )

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


class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config,num_labels,is_IGM,is_LabelEmb,iter_num,LabelEmb_dim):
        super().__init__(config)

        config.num_labels = num_labels
        self.num_labels = config.num_labels
        self.config = config

        self.is_IGM = is_IGM
        self.is_LabelEmb = is_LabelEmb
        print('igm: ', is_IGM)
        print('label emb: ', is_LabelEmb)

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if is_LabelEmb:
            self.LabelEmb_dim = LabelEmb_dim
            self.label_ids = torch.tensor([
                [22930], #support
                [41561], #refute
                [12516]  #neutral
            ]).to('cuda')

        if is_IGM:
            self.classifier = RobertaClassificationHead(config,vec_num=4,is_LabelEmb=is_LabelEmb,embed_dim=LabelEmb_dim)
            self.igm_layer = IGM(dim=config.hidden_size)
            self.iter_num = iter_num
        else:
            self.classifier = RobertaClassificationHead(config,vec_num=3,is_LabelEmb=is_LabelEmb,embed_dim=LabelEmb_dim)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )

    def get_sub_tensor(self,input_ids,tensor,mask,type='evidence'):
        bs = tensor.shape[0]
        lens = mask.sum(dim=1)
        max_len = int(torch.max(lens,dim=-1).values)
        sub_tensor = torch.zeros(bs,max_len,self.config.hidden_size).to(input_ids.device)
        sub_mask = torch.zeros(bs,max_len).to(input_ids.device)
        for i in range(bs):
            s_idxs = torch.nonzero(input_ids[i]==2).squeeze()
            if type == 'claim':
                s,d = int(0), int(s_idxs[0])
            elif type == 'evidence':
                s,d = int(s_idxs[0]),int(s_idxs[1])
            else:
                s,d = int(s_idxs[1]),int(s_idxs[2])
            sub_tensor[i,:lens[i],:] = tensor[i,s+1:d,:]
            sub_mask[i,:lens[i]] = 1
        return sub_tensor, sub_mask

    def get_sub_tensor2(self,input_ids,tensor,mask,type='evidence'):
        sub_tensor = tensor.clone()
        sub_mask = mask.clone()
        sub_tensor[~mask.bool(),:] = 0
        return sub_tensor, sub_mask

    def get_label_feats(self):
        embeddings = self.roberta.embeddings(input_ids=self.label_ids)
        embeddings = embeddings.squeeze(1)
        return embeddings

    def forward(
            self,
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

            claim_mask=None,
            evidence_mask=None,
            question_mask=None,

            label_idx=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] #(bs,len,dim)
        if self.is_IGM == False:
            claim_vector  = torch.einsum("bsh,bs,b->bh", sequence_output, claim_mask.float(), #(bs,dim)
                                         1 / claim_mask.float().sum(dim=1) + 1e-9)

            evidence_vector = torch.einsum("bsh,bs,b->bh", sequence_output, evidence_mask.float(), #(bs,dim)
                                            1 / evidence_mask.float().sum(dim=1) + 1e-9)
            question_vector = torch.einsum("bsh,bs,b->bh", sequence_output, question_mask.float(), #(bs,dim)
                                1 / question_mask.float().sum(dim=1) + 1e-9)
            cat_vector = torch.cat([claim_vector,evidence_vector,question_vector],dim=1)

        else:
            claim_hidden_state,claim_alone_mask = self.get_sub_tensor2(input_ids,sequence_output,claim_mask,type='claim')
            evidence_hidden_state,evidence_alone_mask = self.get_sub_tensor2(input_ids,sequence_output,evidence_mask,type='evidence')
            question_hidden_state,question_alone_mask = self.get_sub_tensor2(input_ids,sequence_output,question_mask,type='question')

            for i in range(self.iter_num):
                evidence_hidden_state_tmp = self.igm_layer(evidence_hidden_state,question_hidden_state,input3=claim_hidden_state,pooling_type='max',
                                                           mask1=evidence_alone_mask,mask2=question_alone_mask,mask3=claim_alone_mask)
                question_hidden_state_tmp = self.igm_layer(question_hidden_state,evidence_hidden_state,input3=claim_hidden_state,pooling_type='max',
                                                           mask1=question_alone_mask,mask2=evidence_alone_mask,mask3=claim_alone_mask)

                evidence_hidden_state = evidence_hidden_state_tmp
                question_hidden_state = question_hidden_state_tmp


            claim_hidden_state1 = self.igm_layer(claim_hidden_state,evidence_hidden_state,pooling_type='max',
                                                 mask1=claim_alone_mask,mask2=evidence_alone_mask)
            claim_vector1  = torch.einsum("bsh,bs,b->bh", claim_hidden_state1, claim_alone_mask.float(), #(bs,dim)
                                          1 / claim_alone_mask.float().sum(dim=1) + 1e-9)
            claim_hidden_state2 = self.igm_layer(claim_hidden_state,question_hidden_state,pooling_type='max',
                                                 mask1=claim_alone_mask,mask2=question_alone_mask)
            claim_vector2  = torch.einsum("bsh,bs,b->bh", claim_hidden_state2, claim_alone_mask.float(), #(bs,dim)
                                          1 / claim_alone_mask.float().sum(dim=1) + 1e-9)

            evidence_vector  = torch.einsum("bsh,bs,b->bh", evidence_hidden_state, evidence_alone_mask.float(), #(bs,dim)
                                            1 / evidence_alone_mask.float().sum(dim=1) + 1e-9)
            question_vector  = torch.einsum("bsh,bs,b->bh", question_hidden_state, question_alone_mask.float(), #(bs,dim)
                                            1 / question_alone_mask.float().sum(dim=1) + 1e-9)

            cat_vector = torch.cat([claim_vector1,claim_vector2,evidence_vector,question_vector],dim=1)

        if self.is_LabelEmb:
            if label_idx is not None:
                if len(label_idx.shape)==2:
                    label_idx = label_idx[0]
                label_feats = sequence_output[:,label_idx,:]
                # label_feats2 = self.get_label_feats()
                # label_feats2 = label_feats2.repeat(sequence_output.shape[0],1,1)
                # label_feats = label_feats * 0.5 + label_feats2 * 0.5
            else:
                label_feats = self.get_label_feats()
                label_feats = label_feats.repeat(sequence_output.shape[0],1,1)

            logits,npy_lf,npy_x = self.classifier(None,x=cat_vector,label_feats=label_feats)

        else:
            logits = self.classifier(None,x=cat_vector,label_feats=None)


        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = {'loss':loss}

            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            # output = (logits,) + outputs[2:]
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        ),npy_lf,npy_x

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, vec_num=1,is_LabelEmb=True,embed_dim=128):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size * vec_num, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        if is_LabelEmb:
            self.embed_dim = embed_dim
            self.embedding_dense = nn.Linear(config.hidden_size, self.embed_dim)
            self.embedding_out_proj = nn.Linear(config.hidden_size + self.embed_dim * config.num_labels, config.num_labels)
            self.emb_attention = SelfAttention(input_dim = config.hidden_size+self.embed_dim*config.num_labels)

    def forward(self, features, x=None, label_feats = None, **kwargs):
        if x is None:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x) # bs * dim
        npy_x = x.data.cpu().numpy()
        if label_feats is not None:
            label_feats = self.dropout(label_feats)
            label_feats = self.embedding_dense(label_feats)
            if self.config.num_labels == 3:
                emb_x = torch.cat([x,label_feats[:,0,:],label_feats[:,1,:],label_feats[:,2,:]],dim=1)
            else:
                emb_x = torch.cat([x,label_feats[:,0,:],label_feats[:,1,:]],dim=1)
            emb_x = self.emb_attention(emb_x.unsqueeze(1))
            npy_lf = emb_x.data.cpu().numpy()
            emb_x = torch.tanh(emb_x)
            emb_x = self.dropout(emb_x)
            emb_x = self.embedding_out_proj(emb_x)
            return emb_x,npy_lf,npy_x

        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x