---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:363858
- loss:CosineSimilarityLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: What are the good video tutorials to learn Java 8?
  sentences:
  - What is the function of a nucleus in animal cells?
  - What is best tutorial/blog where I can learning Java 8?
  - Assembly of parts units?
- source_sentence: What was your most embarrassing moment in front of a doctor?
  sentences:
  - Why does cognizant having so many hierarchy?
  - What was the most embarrassing moment with your cousin?
  - If I score 90-100marks in cat and I am belonging from St cast I have any chance
    for get admission in Nirma university?
- source_sentence: How did you exactly break up from your ex?
  sentences:
  - What is the best way to break with a narcissist?
  - Are Trump's cabinet picks expected or are they all surprises?
  - Why and how did you break up with her/him?
- source_sentence: What are alternatives to healthprofs.com?
  sentences:
  - What are some alternatives to thebetafamily.com?
  - I want to feel beautiful. What do I do?
  - 'Why does the following code produce the output: 0 0 1 2?'
- source_sentence: Could anyone give me some interesting website where I could start
    learning the JSL sign language ?
  sentences:
  - Could anyone give me some interesting website where I could start learning the
    sign language (JSL 日本手話語族)?
  - Is it right to read your girlfriend's text message, given that her phone is unlocked
    and the text message app is already launched?
  - How can you know if you're in love or just attracted to someone?
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: quora val
      type: quora-val
    metrics:
    - type: pearson_cosine
      value: 0.7629059223791574
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.7368675215359358
      name: Spearman Cosine
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'Could anyone give me some interesting website where I could start learning the JSL sign language ?',
    'Could anyone give me some interesting website where I could start learning the sign language (JSL 日本手話語族)?',
    "Is it right to read your girlfriend's text message, given that her phone is unlocked and the text message app is already launched?",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5946, 0.0953],
#         [0.5946, 1.0000, 0.1109],
#         [0.0953, 0.1109, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity

* Dataset: `quora-val`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.7629     |
| **spearman_cosine** | **0.7369** |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 363,858 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                        | label                                                          |
  |:--------|:----------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                            | float                                                          |
  | details | <ul><li>min: 5 tokens</li><li>mean: 15.43 tokens</li><li>max: 62 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 15.71 tokens</li><li>max: 71 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.39</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                            | sentence_1                                                        | label            |
  |:--------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------|:-----------------|
  | <code>Which is better in mobile development - iOS or Android?</code>                                                                  | <code>Is it iOS is better than Android?</code>                    | <code>1.0</code> |
  | <code>What should I do if I don't have any initials in my name but the 'last name' field is mandatory?</code>                         | <code>What was Arrow's last name?</code>                          | <code>0.0</code> |
  | <code>Do recruiters pay recruiting agency in advance the charges of recruiting agency or after the selection of the candidate?</code> | <code>What is the best IT-recruitment agency in Amsterdam?</code> | <code>0.0</code> |
* Loss: [<code>CosineSimilarityLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosinesimilarityloss) with these parameters:
  ```json
  {
      "loss_fct": "torch.nn.modules.loss.MSELoss"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_ratio`: None
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `enable_jit_checkpoint`: False
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `use_cpu`: False
- `seed`: 42
- `data_seed`: None
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: -1
- `ddp_backend`: None
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `auto_find_batch_size`: False
- `full_determinism`: False
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `use_cache`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss | quora-val_spearman_cosine |
|:------:|:-----:|:-------------:|:-------------------------:|
| 0.0220 | 500   | 0.2239        | -                         |
| 0.0440 | 1000  | 0.1783        | -                         |
| 0.0660 | 1500  | 0.1573        | -                         |
| 0.0879 | 2000  | 0.1463        | -                         |
| 0.1099 | 2500  | 0.1402        | -                         |
| 0.1319 | 3000  | 0.1389        | -                         |
| 0.1539 | 3500  | 0.1368        | -                         |
| 0.1759 | 4000  | 0.1330        | -                         |
| 0.1979 | 4500  | 0.1319        | -                         |
| 0.2199 | 5000  | 0.1273        | -                         |
| 0.2418 | 5500  | 0.1255        | -                         |
| 0.2638 | 6000  | 0.1239        | -                         |
| 0.2858 | 6500  | 0.1270        | -                         |
| 0.3078 | 7000  | 0.1245        | -                         |
| 0.3298 | 7500  | 0.1239        | -                         |
| 0.3518 | 8000  | 0.1210        | -                         |
| 0.3738 | 8500  | 0.1186        | -                         |
| 0.3957 | 9000  | 0.1230        | -                         |
| 0.4177 | 9500  | 0.1162        | -                         |
| 0.4397 | 10000 | 0.1140        | -                         |
| 0.4617 | 10500 | 0.1172        | -                         |
| 0.4837 | 11000 | 0.1194        | -                         |
| 0.5057 | 11500 | 0.1152        | -                         |
| 0.5277 | 12000 | 0.1147        | -                         |
| 0.5496 | 12500 | 0.1169        | -                         |
| 0.5716 | 13000 | 0.1147        | -                         |
| 0.5936 | 13500 | 0.1168        | -                         |
| 0.6156 | 14000 | 0.1117        | -                         |
| 0.6376 | 14500 | 0.1133        | -                         |
| 0.6596 | 15000 | 0.1124        | -                         |
| 0.6816 | 15500 | 0.1085        | -                         |
| 0.7035 | 16000 | 0.1123        | -                         |
| 0.7255 | 16500 | 0.1074        | -                         |
| 0.7475 | 17000 | 0.1095        | -                         |
| 0.7695 | 17500 | 0.1087        | -                         |
| 0.7915 | 18000 | 0.1105        | -                         |
| 0.8135 | 18500 | 0.1100        | -                         |
| 0.8355 | 19000 | 0.1109        | -                         |
| 0.8574 | 19500 | 0.1122        | -                         |
| 0.8794 | 20000 | 0.1052        | -                         |
| 0.9014 | 20500 | 0.1097        | -                         |
| 0.9234 | 21000 | 0.1092        | -                         |
| 0.9454 | 21500 | 0.1078        | -                         |
| 0.9674 | 22000 | 0.1059        | -                         |
| 0.9894 | 22500 | 0.1083        | -                         |
| 1.0    | 22742 | -             | 0.7272                    |
| 1.0113 | 23000 | 0.1035        | -                         |
| 1.0333 | 23500 | 0.0940        | -                         |
| 1.0553 | 24000 | 0.0985        | -                         |
| 1.0773 | 24500 | 0.0951        | -                         |
| 1.0993 | 25000 | 0.0959        | -                         |
| 1.1213 | 25500 | 0.0986        | -                         |
| 1.1433 | 26000 | 0.0978        | -                         |
| 1.1652 | 26500 | 0.0969        | -                         |
| 1.1872 | 27000 | 0.0995        | -                         |
| 1.2092 | 27500 | 0.0961        | -                         |
| 1.2312 | 28000 | 0.0942        | -                         |
| 1.2532 | 28500 | 0.0972        | -                         |
| 1.2752 | 29000 | 0.0941        | -                         |
| 1.2972 | 29500 | 0.0927        | -                         |
| 1.3191 | 30000 | 0.0941        | -                         |
| 1.3411 | 30500 | 0.0971        | -                         |
| 1.3631 | 31000 | 0.0965        | -                         |
| 1.3851 | 31500 | 0.0944        | -                         |
| 1.4071 | 32000 | 0.0962        | -                         |
| 1.4291 | 32500 | 0.0967        | -                         |
| 1.4511 | 33000 | 0.0961        | -                         |
| 1.4730 | 33500 | 0.0952        | -                         |
| 1.4950 | 34000 | 0.0943        | -                         |
| 1.5170 | 34500 | 0.0968        | -                         |
| 1.5390 | 35000 | 0.0985        | -                         |
| 1.5610 | 35500 | 0.0971        | -                         |
| 1.5830 | 36000 | 0.0954        | -                         |
| 1.6050 | 36500 | 0.0929        | -                         |
| 1.6269 | 37000 | 0.0954        | -                         |
| 1.6489 | 37500 | 0.0933        | -                         |
| 1.6709 | 38000 | 0.0958        | -                         |
| 1.6929 | 38500 | 0.0948        | -                         |
| 1.7149 | 39000 | 0.0918        | -                         |
| 1.7369 | 39500 | 0.0939        | -                         |
| 1.7589 | 40000 | 0.0924        | -                         |
| 1.7808 | 40500 | 0.0952        | -                         |
| 1.8028 | 41000 | 0.0937        | -                         |
| 1.8248 | 41500 | 0.0948        | -                         |
| 1.8468 | 42000 | 0.0961        | -                         |
| 1.8688 | 42500 | 0.0924        | -                         |
| 1.8908 | 43000 | 0.0935        | -                         |
| 1.9128 | 43500 | 0.0933        | -                         |
| 1.9347 | 44000 | 0.0953        | -                         |
| 1.9567 | 44500 | 0.0919        | -                         |
| 1.9787 | 45000 | 0.0929        | -                         |
| 2.0    | 45484 | -             | 0.7369                    |


### Framework Versions
- Python: 3.10.11
- Sentence Transformers: 5.2.2
- Transformers: 5.1.0
- PyTorch: 2.5.1+cu121
- Accelerate: 1.12.0
- Datasets: 3.5.0
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->