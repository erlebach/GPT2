python src/scripts/evaluate_models.py
Evaluating model on validation dataset...
============================================================
EVALUATION METRICS SUMMARY
============================================================
Perplexity: 50200.7681
Loss: 10.8238
Accuracy: 0.0000
Entropy: 10.8215
Repetition Penalty: 0.0002
Diversity Score: 0.8952

Top-k Accuracy:
  Top-1: 0.0000
  Top-3: 0.0000
  Top-5: 0.0000
============================================================

Generating sample text...
Traceback (most recent call last):
  File "/Users/erlebach/src/2025/GPT2/src/scripts/evaluate_models.py", line 71, in <module>
    main()
  File "/Users/erlebach/src/2025/GPT2/src/scripts/evaluate_models.py", line 58, in main
    generated_text = generate_text_for_evaluation(
  File "<@beartype(utils.evaluation.generate_text_for_evaluation) at 0x12dd249d0>", line 141, in generate_text_for_evaluation
  File "/Users/erlebach/src/2025/GPT2/src/utils/evaluation.py", line 504, in generate_text_for_evaluation
    logits, _ = model(input_ids, None)
  File "/Users/erlebach/src/2025/GPT2/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/Users/erlebach/src/2025/GPT2/.venv/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Users/erlebach/src/2025/GPT2/src/models/gpt2/model.py", line 321, in forward
    T <= self.config.block_size
AssertionError: Cannot forward sequence of length 33, block size is only 32

----------------------------------------------------------------------------------------------------



