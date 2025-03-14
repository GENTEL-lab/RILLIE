import os
from typing import Union, Optional, List, Callable, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_info

from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict

from modelgenerator.backbones.base import *


class GenBioBERT(HFSequenceBackbone):
    """GenBioBERT model

    Note:
        Models using this interface include `aido_dna_7b`, `aido_dna_300m`, `dna_dummy`, `aido_dna_debug`,
        `aido_rna_1b600m`, `aido_rna_1b600m_cds`, `aido_rna_1m_mars`, `aido_rna_25m_mars`, `aido_rna_300m_mars`,
        `aido_rna_650m`, `aido_rna_650m_cds`.

        FSDP auto_wrap_policy is `[transformers.models.rnabert.modeling_rnabert.RNABertLayer]`

    Args:
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 32.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value"].
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = ["query", "value"],
        **kwargs,
    ):
        # Delays hf model imports to avoid model name conflicts
        from modelgenerator.huggingface_models.rnabert import (
            RNABertConfig,
            RNABertTokenizer,
            RNABertModel,
            RNABertForMaskedLM,
            RNABertForTokenClassification,
            RNABertForSequenceClassification,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        repo_base_dir = Path(__file__).resolve().parent.parent.parent
        vocab_file = os.path.join(
            repo_base_dir, "modelgenerator/huggingface_models/rnabert/vocab.txt"
        )
        self.max_length = max_length
        self.tokenizer = RNABertTokenizer(
            vocab_file, version="v2"
        )  # add [CLS] ... [SEP]
        if self.use_legacy_adapter:
            rank_zero_info(
                "You are using a legacy adapter/head, so its configuration has to be "
                "set explicitly under backbone. This is done using "
                "`model.backbone.config_overwrites` and `model.backbone.model_init_args`."
            )
        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = RNABertForSequenceClassification
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = RNABertForTokenClassification
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = RNABertForMaskedLM
        else:
            model_class = RNABertModel

        if from_scratch:
            config = RNABertConfig()
        else:
            config = RNABertConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )

        if not self.use_legacy_adapter:
            self.encoder = model
        else:
            self.encoder = model.bert
        self.decoder = None
        if model_class == RNABertForMaskedLM:
            self.decoder = model.cls
        elif model_class == RNABertForTokenClassification:
            self.decoder = model.classifier
        elif model_class == RNABertForSequenceClassification:
            self.decoder = model.classifier

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        self.max_length = max_length
        if max_length is None:
            rank_zero_info(
                "You didn't set a max_length for the data in the downstream task"
            )
        if use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                modules_to_save=[],
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if all_hidden_states:
            return outputs.hidden_states
        return outputs.last_hidden_state

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            padding=padding,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=self.max_length is not None,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_mask = torch.tensor(
            seq_tokenized["special_tokens_mask"], dtype=torch.bool
        )
        return input_ids, attention_mask, special_mask

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the pre-trained model

        Returns:
            int: Vocabulary size
        """
        return self.encoder.config.vocab_size

    def on_save_checkpoint(self, checkpoint: dict):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
            prefixed_dict = {f"backbone.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith("backbone.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class GenBioFM(HFSequenceBackbone):
    """GenBioFM model

    Note:
        Models using this interface include `aido_protein_16b`, `aido_protein_16b_v1`, `aido_protein2structoken_16b`, `aido_protein_debug`.

        FSDP auto_wrap_policy is `[modelgenerator.huggingface_models.fm4bio.modeling_fm4bio.FM4BioLayer]`

    Args:
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value", "key", "dense", "router"].
        lora_modules_to_save (Optional[List[str]], optional): LoRA modules to save. Defaults to None.
        lora_use_rslora (bool, optional): Whether to use RSLora. Defaults to False.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = [
            "query",
            "value",
            "key",
            "dense",
            "router",
        ],
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        **kwargs,
    ):
        from modelgenerator.huggingface_models.fm4bio import (
            FM4BioConfig,
            FM4BioTokenizer,
            FM4BioModel,
            FM4BioForMaskedLM,
            FM4BioForTokenClassification,
            FM4BioForSequenceClassification,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.max_length = max_length
        if self.use_legacy_adapter:
            rank_zero_info(
                "You are using a legacy adapter/head, so its configuration has to be "
                "set explicitly under backbone. This is done using "
                "`model.backbone.config_overwrites` and `model.backbone.model_init_args`."
            )
        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = FM4BioForSequenceClassification
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = FM4BioForTokenClassification
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = FM4BioForMaskedLM
        else:
            model_class = FM4BioModel

        repo_base_dir = Path(__file__).resolve().parent.parent.parent
        vocab_file = os.path.join(
            repo_base_dir, "modelgenerator/huggingface_models/fm4bio/vocab_protein.txt"
        )
        self.tokenizer = FM4BioTokenizer(vocab_file, version="v1")

        if from_scratch:
            config = FM4BioConfig()
        else:
            config = FM4BioConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )

        if not self.use_legacy_adapter:
            self.encoder = model
        else:
            self.encoder = model.bert
        self.decoder = None
        if model_class == FM4BioForMaskedLM:
            try:
                self.decoder = model.output_embed
            except AttributeError:
                self.decoder = model.cls
        elif model_class == FM4BioForTokenClassification:
            self.decoder = model.classifier
        elif model_class == FM4BioForSequenceClassification:
            self.decoder = model.classifier

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        if use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:  # use linear probing, freeze all parameters
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        if all_hidden_states:
            return outputs.hidden_states
        return outputs.last_hidden_state

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=padding,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_mask = torch.tensor(
            seq_tokenized["special_tokens_mask"], dtype=torch.bool
        )
        return input_ids, attention_mask, special_mask

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the pre-trained model

        Returns:
            int: Vocabulary size
        """
        return self.encoder.config.vocab_size

    def on_save_checkpoint(self, checkpoint: dict):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
            prefixed_dict = {f"backbone.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith("backbone.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class GenBioCellFoundation(HFSequenceBackbone):
    """GenBioCellFoundation model

    Note:
        Models using this interface include `aido_cell_100m`, `aido_cell_10m`, and `aido_cell_3m`.

        FSDP auto_wrap_policy is `[modelgenerator.huggingface_models.cellfoundation.modeling_cellfoundation.CellFoundationLayer]`

    Args:
        config_overwrites (dict, optional): Optional model arguments for PretrainedConfig. Defaults to None.
        model_init_args (dict, optional): Optional model arguments passed to its init method. Defaults to None.
        from_scratch (bool, optional): Whether to create the model from scratch. Defaults to False.
        max_length (int, optional): Maximum sequence length. Defaults to 512.
        use_peft (bool, optional): Whether to use LoRA PEFT. Defaults to False.
        frozen (bool, optional): Whether to freeze encoder. Defaults to False.
        save_peft_only (bool, optional): Whether to save only the PEFT weights. Defaults to True.
        lora_r (int, optional): LoRA r parameter. Defaults to 16.
        lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.1.
        lora_target_modules (Optional[List[str]], optional): LoRA target modules. Defaults to ["query", "value", "key", "dense", "router"].
        lora_modules_to_save (Optional[List[str]], optional): LoRA modules to save. Defaults to None.
        lora_use_rslora (bool, optional): Whether to use RSLora. Defaults to False.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],  # Should not need this.
        default_config: Union[DefaultConfig, None],
        from_scratch: bool = False,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        frozen: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = [
            "query",
            "value",
            "key",
            "dense",
            "router",
        ],
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        **kwargs,
    ):
        from modelgenerator.huggingface_models.cellfoundation import (
            CellFoundationConfig,
            CellFoundationModel,
        )

        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.max_length = max_length
        # Note: Legacy adapters are for older sequence models.
        if legacy_adapter_type is not None:
            raise NotImplementedError(
                "Legacy adapters are not implemented for CellFoundation."
            )
        model_class = CellFoundationModel
        peft_task_type = TaskType.FEATURE_EXTRACTION

        if from_scratch:
            config = CellFoundationConfig()
        else:
            config = CellFoundationConfig.from_pretrained(self.model_path)
        for k, v in self.config_overwrites.items():
            setattr(config, k, v)

        if from_scratch:
            model = model_class(config=config, **self.model_init_args)
        else:
            model = model_class.from_pretrained(
                self.model_path, config=config, **self.model_init_args
            )
        self.encoder = model
        self.decoder = None

        self.use_peft = use_peft
        self.frozen = frozen
        self.save_peft_only = save_peft_only
        if use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            rank_zero_only(self.encoder.print_trainable_parameters)()
        else:
            if frozen:
                rank_zero_info("> Use Linear Probing. The encoder is frozen.")
                for name, param in self.encoder.named_parameters():
                    param.requires_grad = False

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, all_hidden_states: bool = False
    ) -> Union[Tensor, list]:
        """Encoder-only forward pass

        Args:
            input_ids (torch.Tensor): Input token IDs (n, seq_len)
            attention_mask (torch.Tensor): Attention mask (n, seq_len)
            all_hidden_states (bool, optional): Whether to return all hidden states. Defaults to False.

        Returns:
            Union[Tensor, list]: Last hidden state or list of all hidden states or logits
        """
        X = torch.tensor(
            input_ids, dtype=torch.bfloat16
        )  # Converting from torch.long; should be counts.

        # https://github.com/fm4bio/scFoundation-repro/blob/9f706d807b68ec7b7f2df735d2a96fb4b1b67a0c/annotation/cell_annotation.py#L126-L141:
        rawcountsidx = max(torch.log10(X.sum()), 5)
        inputcountidx = max(torch.log10(X.sum()), 5)
        X = torch.log1p(X / X.sum() * 10000).to(torch.float)
        X = torch.cat(
            (
                X,
                torch.tensor([rawcountsidx, inputcountidx])
                .repeat(X.shape[0], 1)
                .to(X.device),
            ),
            axis=1,
        ).float()
        X[X > 20] = 20

        outputs = self.encoder(
            input_ids=X,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Note: Trimming off embeddings corresponding to read depth inputs.
        if all_hidden_states:
            return (x[:, :-2, :] for x in outputs.hidden_states)
        return outputs.last_hidden_state[:, :-2, :]

    def get_decoder(self) -> nn.Module:
        """Returns the pre-trained decoder

        Returns:
            nn.Module: Decoder
        """
        return self.decoder

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Note:
            This is a dummy tokenizer since the CellFoundation models consume gene expression.

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
        """
        input_ids = sequences
        attention_mask = torch.ones_like(sequences)
        special_tokens_mask = torch.ones_like(sequences)
        return input_ids, attention_mask, special_tokens_mask

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def get_max_context(self) -> int:
        """Returns the maximum context length of the pre-trained model

        Returns:
            int: Maximum context length
        """
        return self.encoder.config.max_position_embeddings

    def get_embedding_size(self) -> int:
        """Returns the hidden size of the pre-trained model

        Returns:
            int: Hidden size
        """
        return self.encoder.config.hidden_size

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Not implemented for CellFoundation.")

    def on_save_checkpoint(self, checkpoint: dict):
        if (not self.use_peft and not self.frozen) or not self.save_peft_only:
            return
        adapter_name = "default"
        if self.use_peft:
            peft_dict = get_peft_model_state_dict(
                self.encoder, adapter_name=adapter_name
            )
            prefixed_dict = {f"backbone.encoder.{k}": v for k, v in peft_dict.items()}
        for k in list(checkpoint["state_dict"].keys()):
            if not k.startswith("backbone.encoder."):
                # keep all decoder weights
                continue
            if self.frozen or (
                self.use_peft
                and k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)

    def get_num_layer(self) -> int:
        """Returns the number of attention layer in the pre-trained model

        Returns:
            int: the number of attention layer
        """
        return self.encoder.config.num_hidden_layers


class Onehot(HFSequenceBackbone):
    """Tokenizer-only model for one-hot encoding. Useful for baseline model testing (CNNs, linear, etc.)

    Note:
        Models using this interface include `dna_onehot` and `protein_onehot`.

        Does not contain any parameters, and cannot be used without an adapter.

    Args:
        vocab_file (str, optional): Path to the vocabulary file. Defaults to "DNA-Transformers/src/transformers/models/rnabert/vocab.txt".
        max_length (Optional[int], optional): Maximum sequence length. Defaults to 512.
    """

    vocab_file: str = os.path.join(
        Path(__file__).resolve().parent.parent.parent,
        "modelgenerator/huggingface_models/rnabert/vocab.txt",
    )

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        vocab_file: Optional[str] = None,
        max_length: Optional[int] = 512,
        **kwargs,
    ):
        from modelgenerator.huggingface_models.fm4bio import FM4BioTokenizer

        super().__init__(None, None, **kwargs)
        self.max_length = max_length
        if vocab_file is not None:
            self.vocab_file = vocab_file
        self.tokenizer = FM4BioTokenizer(self.vocab_file, version="v1")

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Returns one-hot encoding of input_ids.

        Args:
            input_ids (Tensor): Token IDs
            attention_mask (Tensor): Attention mask

        Returns:
            Tensor: One-hot encoding of input_ids
        """
        one_hot = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.tokenizer.vocab_size,
        ).to(input_ids.device)
        one_hot.scatter_(2, input_ids.unsqueeze(2), 1)
        return one_hot

    def get_decoder(self) -> nn.Module:
        """Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return nn.Identity()

    def tokenize(
        self,
        sequences: list[str],
        padding: bool = True,
        add_special_tokens: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor, Tensor]: Token IDs with padding and special tokens, attention mask, and special tokens mask.
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=True,
            padding=padding,
            max_length=self.max_length,
            add_special_tokens=add_special_tokens,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_mask = torch.tensor(
            seq_tokenized["special_tokens_mask"], dtype=torch.bool
        )
        return input_ids, attention_mask, special_mask

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the arbitrary max context specified by the user

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def on_save_checkpoint(self, checkpoint: dict):
        return

    def get_num_layer(self) -> int:
        return


class Huggingface(HFSequenceBackbone):
    """A generic huggingface wrapper allows for using any huggingface model as backbone.

        Warning: This is an experimental feature, don't expect it to work with all models.
        Downstream task support is also extremely limited to the standard huggingface heads.
        Its usage often involves manual configuration of the model's head through `config_overwrites`.

    Args:
        model_path (str): Path to the huggingface model
        max_length (int, optional): Maximum sequence length. Defaults to None.
    """

    def __init__(
        self,
        legacy_adapter_type: Union[LegacyAdapterType, None],
        default_config: Union[DefaultConfig, None],
        model_path: str | os.PathLike,
        modules_for_model_registration: Optional[List[str]] = None,
        max_length: Optional[int] = None,
        use_peft: bool = False,
        save_peft_only: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        lora_modules_to_save: Optional[List[str]] = None,
        lora_use_rslora: bool = False,
        **kwargs,
    ):
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForMaskedLM,
            AutoModelForTokenClassification,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
        import importlib

        if legacy_adapter_type is None:
            raise ValueError(
                "Huggingface models can only be used with legacy adapters."
            )
        modules_for_model_registration = modules_for_model_registration or []
        for module in modules_for_model_registration:
            importlib.import_module(module)
        super().__init__(legacy_adapter_type, default_config, **kwargs)
        self.model_path = model_path
        self.max_length = max_length
        self.use_peft = use_peft
        self.save_peft_only = save_peft_only
        if self.legacy_adapter_type is LegacyAdapterType.SEQ_CLS:
            model_class = AutoModelForSequenceClassification
            peft_task_type = TaskType.SEQ_CLS
        elif self.legacy_adapter_type is LegacyAdapterType.TOKEN_CLS:
            model_class = AutoModelForTokenClassification
            peft_task_type = TaskType.TOKEN_CLS
        elif self.legacy_adapter_type is LegacyAdapterType.MASKED_LM:
            model_class = AutoModelForMaskedLM
            peft_task_type = TaskType.FEATURE_EXTRACTION
        elif self.legacy_adapter_type is None:
            model_class = AutoModel
            peft_task_type = TaskType.FEATURE_EXTRACTION
        else:
            raise ValueError(
                f"There is no standard huggingface head for the task type: {self.legacy_adapter_type}. "
                "Please create a backbone for your huggingfce model."
            )
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        def nested_set_config(config, config_overwrites):
            for k, v in config_overwrites.items():
                if isinstance(v, dict):
                    nested_set_config(getattr(config, k), v)
                else:
                    setattr(config, k, v)

        nested_set_config(config, self.config_overwrites)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, config=config, trust_remote_code=True
        )
        self.model, self.loading_info = model_class.from_pretrained(
            self.model_path,
            config=config,
            trust_remote_code=True,
            output_loading_info=True,
            **self.model_init_args,
        )
        if use_peft:
            peft_config = LoraConfig(
                task_type=peft_task_type,
                target_modules=lora_target_modules,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=lora_use_rslora,
                inference_mode=False,
                modules_to_save=lora_modules_to_save,
            )
            self.model = get_peft_model(self.model, peft_config)
            rank_zero_only(self.model.print_trainable_parameters)()

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Returns the final logits.

        Args:
            input_ids (Tensor): Token IDs
            attention_mask (Tensor): Attention mask

        Returns:
            Tensor: Logits
        """
        return self.model(input_ids, attention_mask).logits

    def get_decoder(self) -> nn.Module:
        """Returns a dummy pass-through decoder

        Returns:
            nn.Module: Decoder
        """
        return nn.Identity()

    def tokenize(self, sequences: list[str]) -> tuple[Tensor, Tensor]:
        """Tokenizes a list of sequences

        Args:
            sequences (list[str]): List of sequences

        Returns:
            tuple[Tensor, Tensor]: Token IDs with padding and special tokens, and attention mask
        """
        seq_tokenized = self.tokenizer(
            sequences,
            truncation=self.max_length is not None,
            padding=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        input_ids = seq_tokenized["input_ids"]
        attention_mask = seq_tokenized["attention_mask"]
        special_tokens_mask = torch.tensor(
            seq_tokenized["special_tokens_mask"], dtype=torch.bool
        )
        return input_ids, attention_mask, special_tokens_mask

    def decode_tokens(self, tokenized_sequences: Tensor) -> list[str]:
        """Decodes a Tensor of token id sequences

        Args:
            tokenized_sequences (Tensor): Tokenized sequences

        Returns:
            list[str]: List of decoded sequences
        """
        return self.tokenizer.batch_decode(tokenized_sequences)

    def get_token_id(self, token: str) -> int:
        """Returns the index of a token in the vocabulary

        Args:
            token (str): Token

        Returns:
            int: Token id
        """
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_max_context(self) -> int:
        """Returns the arbitrary max context specified by the user

        Returns:
            int: Maximum context length
        """
        return self.max_length

    def get_embedding_size(self) -> int:
        """Returns the embedding size

        Returns:
            int: Embedding size
        """
        raise NotImplementedError

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size

        Returns:
            int: Vocabulary size
        """
        return self.tokenizer.vocab_size

    def on_save_checkpoint(self, checkpoint: dict):
        if not self.use_peft or not self.save_peft_only:
            return
        adapter_name = "default"
        peft_dict = get_peft_model_state_dict(self.model, adapter_name=adapter_name)
        prefixed_dict = {f"backbone.model.{k}": v for k, v in peft_dict.items()}
        head_keys = tuple(self.loading_info["missing_keys"])
        for k in list(checkpoint["state_dict"].keys()):
            if k.endswith(head_keys):
                # keep all newly added weights
                continue
            if (
                k.replace(f".{adapter_name}", "") not in prefixed_dict
                and k not in prefixed_dict
            ):
                # get_peft_model_state_dict may or may not remove the adapter name
                checkpoint["state_dict"].pop(k)
