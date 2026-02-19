---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:363858
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 512 tokens
- **Number of Output Labels:** 1 label
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of texts
pairs = [
    ["Girls stare at me but they don't approach me. Why is this so?", "What should I do if I notice a girl staring at me in the library and I don't know her?"],
    ['How do deaf people communicate?', 'How do deaf people think?'],
    ['What are some good Malayalam books to read?', 'What are the must read books in Malayalam literature?'],
    ['Why is the Mona Lisa so famous?', 'Why is the Mona Lisa the most famous painting in the world?'],
    ['What determines the value of the currency?', 'How are currency prices determined?'],
]
scores = model.predict(pairs)
print(scores.shape)
# (5,)

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    "Girls stare at me but they don't approach me. Why is this so?",
    [
        "What should I do if I notice a girl staring at me in the library and I don't know her?",
        'How do deaf people think?',
        'What are the must read books in Malayalam literature?',
        'Why is the Mona Lisa the most famous painting in the world?',
        'How are currency prices determined?',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
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
  |         | sentence_0                                                                                      | sentence_1                                                                                      | label                                                          |
  |:--------|:------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                                          | string                                                                                          | float                                                          |
  | details | <ul><li>min: 17 characters</li><li>mean: 57.59 characters</li><li>max: 289 characters</li></ul> | <ul><li>min: 11 characters</li><li>mean: 60.21 characters</li><li>max: 484 characters</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.35</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                 | sentence_1                                                                                          | label            |
  |:---------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Girls stare at me but they don't approach me. Why is this so?</code> | <code>What should I do if I notice a girl staring at me in the library and I don't know her?</code> | <code>0.0</code> |
  | <code>How do deaf people communicate?</code>                               | <code>How do deaf people think?</code>                                                              | <code>0.0</code> |
  | <code>What are some good Malayalam books to read?</code>                   | <code>What are the must read books in Malayalam literature?</code>                                  | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 2

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
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.0220 | 500   | 0.7769        |
| 0.0440 | 1000  | 0.4945        |
| 0.0660 | 1500  | 0.4304        |
| 0.0879 | 2000  | 0.4154        |
| 0.1099 | 2500  | 0.4097        |
| 0.1319 | 3000  | 0.3812        |
| 0.1539 | 3500  | 0.3768        |
| 0.1759 | 4000  | 0.3754        |
| 0.1979 | 4500  | 0.3704        |
| 0.2199 | 5000  | 0.3594        |
| 0.2418 | 5500  | 0.3435        |
| 0.2638 | 6000  | 0.3462        |
| 0.2858 | 6500  | 0.3467        |
| 0.3078 | 7000  | 0.3348        |
| 0.3298 | 7500  | 0.3237        |
| 0.3518 | 8000  | 0.3189        |
| 0.3738 | 8500  | 0.3237        |
| 0.3957 | 9000  | 0.3130        |
| 0.4177 | 9500  | 0.3131        |
| 0.4397 | 10000 | 0.3140        |
| 0.4617 | 10500 | 0.3091        |
| 0.4837 | 11000 | 0.3107        |
| 0.5057 | 11500 | 0.3138        |
| 0.5277 | 12000 | 0.3063        |
| 0.5496 | 12500 | 0.2968        |
| 0.5716 | 13000 | 0.3062        |
| 0.5936 | 13500 | 0.3017        |
| 0.6156 | 14000 | 0.3065        |
| 0.6376 | 14500 | 0.2857        |
| 0.6596 | 15000 | 0.2843        |
| 0.6816 | 15500 | 0.2950        |
| 0.7035 | 16000 | 0.2809        |
| 0.7255 | 16500 | 0.2992        |
| 0.7475 | 17000 | 0.2881        |
| 0.7695 | 17500 | 0.2861        |
| 0.7915 | 18000 | 0.2825        |
| 0.8135 | 18500 | 0.2870        |
| 0.8355 | 19000 | 0.2891        |
| 0.8574 | 19500 | 0.2818        |
| 0.8794 | 20000 | 0.2885        |
| 0.9014 | 20500 | 0.2869        |
| 0.9234 | 21000 | 0.2811        |
| 0.9454 | 21500 | 0.2795        |
| 0.9674 | 22000 | 0.2800        |
| 0.9894 | 22500 | 0.2756        |
| 1.0113 | 23000 | 0.2658        |
| 1.0333 | 23500 | 0.2422        |
| 1.0553 | 24000 | 0.2445        |
| 1.0773 | 24500 | 0.2493        |
| 1.0993 | 25000 | 0.2397        |
| 1.1213 | 25500 | 0.2487        |
| 1.1433 | 26000 | 0.2500        |
| 1.1652 | 26500 | 0.2490        |
| 1.1872 | 27000 | 0.2471        |
| 1.2092 | 27500 | 0.2370        |
| 1.2312 | 28000 | 0.2611        |
| 1.2532 | 28500 | 0.2391        |
| 1.2752 | 29000 | 0.2441        |
| 1.2972 | 29500 | 0.2471        |
| 1.3191 | 30000 | 0.2354        |
| 1.3411 | 30500 | 0.2477        |
| 1.3631 | 31000 | 0.2397        |
| 1.3851 | 31500 | 0.2289        |
| 1.4071 | 32000 | 0.2477        |
| 1.4291 | 32500 | 0.2480        |
| 1.4511 | 33000 | 0.2476        |
| 1.4730 | 33500 | 0.2369        |
| 1.4950 | 34000 | 0.2434        |
| 1.5170 | 34500 | 0.2483        |
| 1.5390 | 35000 | 0.2350        |
| 1.5610 | 35500 | 0.2424        |
| 1.5830 | 36000 | 0.2406        |
| 1.6050 | 36500 | 0.2425        |
| 1.6269 | 37000 | 0.2406        |
| 1.6489 | 37500 | 0.2357        |
| 1.6709 | 38000 | 0.2397        |
| 1.6929 | 38500 | 0.2446        |
| 1.7149 | 39000 | 0.2475        |
| 1.7369 | 39500 | 0.2440        |
| 1.7589 | 40000 | 0.2380        |
| 1.7808 | 40500 | 0.2497        |
| 1.8028 | 41000 | 0.2318        |
| 1.8248 | 41500 | 0.2320        |
| 1.8468 | 42000 | 0.2445        |
| 1.8688 | 42500 | 0.2388        |
| 1.8908 | 43000 | 0.2361        |
| 1.9128 | 43500 | 0.2398        |
| 1.9347 | 44000 | 0.2400        |
| 1.9567 | 44500 | 0.2370        |
| 1.9787 | 45000 | 0.2234        |


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