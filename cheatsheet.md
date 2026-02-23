# Cheatsheet

## Tensor Operations etc

### [torch.where](https://docs.pytorch.org/docs/stable/generated/torch.where.html)

`torch.where(condition, input, other)`

Returns a tensor of elements selected from either `input` (if `condition` is satisfied) or `other` (if `condition` is not satisfied), depending on condition.

```python3
>>> x = torch.tensor([[1, 2, 3], [4, 5, 6]])

>>> threshold = 3

>>> torch.where(x <= threshold, 0, x)
tensor([[0, 0, 0],
        [4, 5, 6]])

>>> torch.where(x > threshold, x - 1, x)
tensor([[1, 2, 3],
        [3, 4, 5]])  
```

## Model Stuff

- [Saving / Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference)
