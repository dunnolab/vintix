# How to use Vintix

## Model Initialization
To initialize a model, you need three checkpoint files:

- `model.pth`: the model’s weights. 
- `config.json`: defines model parameters (e.g., `transformer_depth`, `transformer_heads`, `hidden_dim`) required to set up the model correctly. 
- `metadata.json`: contains task-related information (e.g., `group_name`, `reward_scale`).

Use the following code to initialize and load the checkpoint:

```python
PATH_TO_CHECKPOINT = "/path/to/checkpoint"
model = Vintix()
model.load_model(PATH_TO_CHECKPOINT)
model.to(torch.device('cuda'))
model.eval()
```

## Preparing model
Before starting the sequential prediction, it’s important to prepare the model using the `reset_model` method. This method clears the current context and prepares a new one. You can also pass a prompt using this method and choose the prediction method: with or without KV-cache.

The example of predicting without prompt can be found at [eval_with_empty_context.py](eval_with_empty_context.py) file, the example of predicting with prompt can be found at [eval_with_empty_context.py](eval_with_context.py) file.

## Action predicting
To predict the next action, use the `get_next_action` method. This method takes the current observation and the previous reward, then returns the current action. All required context updates are handled within this method.

## Advanced usage
You can use the `create_model_input`, `get_action`, and `init_cache` methods to write your own evaluation code:

1) `init_cache` returns an empty initialized KV-cache
2) `create_model_input` takes the sequences of observations, within-episode step numbers, previous actions, and previous rewards, then returns the appropriate input batch
3) `get_action` takes the input batch and the KV-cache, and returns the action